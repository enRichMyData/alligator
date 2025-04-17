import os
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.operations import UpdateOne

from alligator import PROJECT_ROOT
from alligator.database import DatabaseAccessMixin
from alligator.feature import DEFAULT_FEATURES
from alligator.mongo import MongoWrapper

if TYPE_CHECKING:
    from tensorflow.keras.models import Model


class MLWorker(DatabaseAccessMixin):
    def __init__(
        self,
        worker_id: int,
        table_name: str,
        dataset_name: str,
        stage: str,
        model_path: str | None = None,
        batch_size: int = 100,
        max_candidates_in_result: int = 5,
        top_n_for_type_freq: int = 3,
        features: List[str] | None = None,
        **kwargs,
    ) -> None:
        super(MLWorker, self).__init__()
        self.worker_id = worker_id
        self.table_name = table_name
        self.dataset_name = dataset_name
        if stage.lower() not in {"rank", "rerank"}:
            raise ValueError(f"Invalid stage: {stage}. Possible values are: 'rank', 'rerank'")
        self.stage = stage
        self.model_path: str = model_path or os.path.join(
            PROJECT_ROOT, "alligator", "models", "default.h5"
        )
        self.batch_size: int = batch_size
        self.max_candidates_in_result: int = max_candidates_in_result
        self.top_n_for_type_freq: int = top_n_for_type_freq
        self.selected_features = features or DEFAULT_FEATURES
        self._db_name = kwargs.pop("db_name", "alligator_db")
        self._mongo_uri = kwargs.pop("mongo_uri", "mongodb://gator-mongodb:27017/")
        self.input_collection = kwargs.get("input_collection", "input_data")
        self.error_logs_collection = kwargs.get("error_collection", "error_logs")
        self.mongo_wrapper: MongoWrapper = MongoWrapper(
            self._mongo_uri,
            self._db_name,
            error_log_collection=self.error_logs_collection,
        )

    def load_ml_model(self) -> "Model":
        from tensorflow.keras.models import load_model  # Local import as in original code

        return load_model(self.model_path)

    def run(self, global_frequencies: Tuple[Dict[Any, Counter], Dict[Any, Counter]] = (None, None)) -> None:
        """Process candidates directly from input_collection"""
        db: Database = self.get_db()
        model: "Model" = self.load_ml_model()
        input_collection: Collection = db[self.input_collection]

        # Unpack type and predicate frequencies
        type_frequencies, predicate_frequencies = global_frequencies
        if type_frequencies is None:
            type_frequencies = {}
        if predicate_frequencies is None:
            predicate_frequencies = {}

        # Now proceed with processing documents in batches
        total_docs = self.mongo_wrapper.count_documents(input_collection, self._get_query())

        processed_count = 0
        while processed_count < total_docs:
            print(
                f"ML ranking for stage {self.stage} progress: "
                f"{processed_count}/{total_docs} documents"
            )

            # Process a batch using the pre-computed global frequencies
            docs_processed = self.apply_ml_ranking(model, type_frequencies, predicate_frequencies)
            processed_count += docs_processed

            # If no documents processed, check if there are any left
            if docs_processed == 0:
                remaining = self.mongo_wrapper.count_documents(input_collection, self._get_query())
                if remaining == 0:
                    break

        print(
            f"ML ranking for stage {self.stage} complete: {processed_count}/{total_docs} documents"
        )

    def apply_ml_ranking(
        self, 
        model: "Model", 
        type_frequencies: Dict[Any, Counter] = {}, 
        predicate_frequencies: Dict[Any, Counter] = {}
    ) -> int:
        """Apply ML ranking using pre-computed global type and predicate frequencies"""
        db: Database = self.get_db()
        input_collection: Collection = db[self.input_collection]

        # 1) Claim a batch of documents to process
        batch_docs = []
        for _ in range(self.batch_size):
            doc = input_collection.find_one_and_update(
                self._get_query(),
                {"$set": {f"{self.stage}_status": "DOING"}},
                projection={"_id": 1, "row_id": 1, "candidates": 1},
            )
            if doc is None:
                break
            batch_docs.append(doc)

        if not batch_docs:
            return 0

        # 2) Assign global frequencies to each candidate, extract features, etc.
        all_candidates = []
        doc_info = []
        for doc in batch_docs:
            row_id = doc["row_id"]
            candidates_by_column: Dict[Any, List[Dict[str, Any]]] = doc["candidates"]
            for col_idx, candidates in candidates_by_column.items():
                # Get frequencies for this column
                cta_counter = type_frequencies.get(col_idx, Counter())
                cpa_counter = predicate_frequencies.get(col_idx, Counter())
                
                for idx, cand in enumerate(candidates):
                    cand_feats = cand.setdefault("features", {})
                    
                    # Process CTA (type frequencies)
                    qids = [t.get("id") for t in cand.get("types", []) if t.get("id")]
                    type_freq_list = sorted([cta_counter.get(qid, 0.0) for qid in qids], reverse=True)

                    # Assign typeFreq1..typeFreq5
                    for i in range(1, 6):
                        cand_feats[f"cta_t{i}"] = (
                            type_freq_list[i - 1] if (i - 1) < len(type_freq_list) else 0.0
                        )
                    
                    # Process CPA (predicate frequencies)
                    pred_scores = {}
                    for rel_col, predicates in cand.get("predicates", {}).items():
                        for pred_id, pred_value in predicates.items():
                            pred_freq = cpa_counter.get(pred_id, 0.0)
                            pred_scores[pred_id] = pred_freq * pred_value
                    
                    # Sort and take top 5
                    pred_freq_list = sorted(pred_scores.values(), reverse=True)
                    
                    # Assign predFreq1..predFreq5
                    for i in range(1, 6):
                        cand_feats[f"cpa_t{i}"] = (
                            pred_freq_list[i - 1] if (i - 1) < len(pred_freq_list) else 0.0
                        )

                    # Build feature vector for ML model
                    feat_vec = self.extract_features(cand)
                    all_candidates.append(feat_vec)
                    doc_info.append((doc["_id"], row_id, col_idx, idx))

        # 3) If no candidates, mark these docs as 'ML_DONE'
        if not all_candidates:
            input_collection.update_many(
                {"_id": {"$in": [d["_id"] for d in batch_docs]}},
                {"$set": {f"{self.stage}_status": "ML_DONE"}},
            )
            return len(batch_docs)

        # 4) ML predictions
        features_array = np.array(all_candidates)
        ml_scores = model.predict(features_array, batch_size=128)[:, 1]

        # 5) Assign scores and prepare updates
        docs_by_id = {doc["_id"]: doc for doc in batch_docs}
        score_map: Dict[Any, Dict[Any, Dict[int, float]]] = {}
        for i, (doc_id, row_id, col_idx, cand_idx) in enumerate(doc_info):
            score_map.setdefault(doc_id, {}).setdefault(col_idx, {})[cand_idx] = float(
                ml_scores[i]
            )

        bulk_updates = []
        for doc_id, doc in docs_by_id.items():
            el_results = {}
            candidates_to_save = {}
            candidates_by_column = doc["candidates"]

            for col_idx, cdict in score_map.get(doc_id, {}).items():
                col_cands = candidates_by_column[col_idx]
                # Update candidate scores
                for c_idx, scr in cdict.items():
                    col_cands[c_idx]["score"] = scr

                # Sort + slice top N
                sorted_cands = sorted(col_cands, key=lambda x: x.get("score", 0.0), reverse=True)
                if self.stage == "rerank" and self.max_candidates_in_result > 0:
                    cands_to_save = sorted_cands[: self.max_candidates_in_result]
                else:
                    cands_to_save = sorted_cands
                el_results[col_idx] = cands_to_save
                candidates_to_save[col_idx] = sorted_cands

            set_query = {f"{self.stage}_status": "DONE", "candidates": candidates_to_save}
            if self.stage == "rerank":
                set_query["el_results"] = el_results
            bulk_updates.append(UpdateOne({"_id": doc_id}, {"$set": set_query}))

        # 6) Bulk commit final results
        if bulk_updates:
            input_collection.bulk_write(bulk_updates)

        return len(batch_docs)

    def _get_query(self) -> Dict[str, Any]:
        query = {
            "dataset_name": self.dataset_name,
            "table_name": self.table_name,
            "status": "DONE",
        }
        if self.stage == "rank":
            query["rank_status"] = "TODO"
        else:
            query["rerank_status"] = "TODO"
        return query

    def extract_features(self, candidate: Dict[str, Any]) -> List[float]:
        """Extract features as in the original system"""
        return [candidate["features"].get(feature, 0.0) for feature in self.selected_features]
