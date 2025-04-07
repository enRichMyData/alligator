import hashlib
import traceback
from collections import defaultdict

import pandas as pd

from alligator.feature import Feature
from alligator.fetchers import CandidateFetcher, LiteralFetcher, ObjectFetcher
from alligator.mongo import MongoWrapper
from alligator.utils import tokenize_text


class RowBatchProcessor:
    """
    Extracted logic for process_rows_batch (and associated scoring helpers).
    Takes the Alligator instance so we can reference .mongo_wrapper, .feature, etc.
    """

    def __init__(
        self,
        feature: Feature,
        candidate_fetcher: CandidateFetcher,
        object_fetcher: ObjectFetcher = None,
        literal_fetcher: LiteralFetcher = None,
        max_candidates_in_result: int = 5,
        **kwargs,
    ):
        self.feature = feature
        self.candidate_fetcher = candidate_fetcher
        self.object_fetcher = object_fetcher
        self.literal_fetcher = literal_fetcher
        self.max_candidates_in_result = max_candidates_in_result
        self._db_name = kwargs.get("db_name", "alligator_db")
        self._mongo_uri = kwargs.get("mongo_uri", "mongodb://gator-mongodb:27017")
        self.input_collection = kwargs.get("input_collection", "input_data")
        self.mongo_wrapper = MongoWrapper(self._mongo_uri, self._db_name)

    def get_db(self):
        """Get MongoDB database connection for current process"""
        from alligator.mongo import MongoConnectionManager

        client = MongoConnectionManager.get_client(self._mongo_uri)
        return client[self._db_name]

    async def process_rows_batch(self, docs):
        """
        Orchestrates the overall flow:
          1) Collect all entities from the batch for candidate fetching.
          2) Fetch initial candidates (batch).
          3) Attempt fuzzy retry if needed.
          4) Enhance with LAMAPI features if configured.
          5) Save results and update DB.
        """
        db = self.get_db()
        try:
            # 1) Gather all needed info from docs
            (
                all_entities_to_process,
                all_row_texts,
                all_fuzzies,
                all_qids,
                row_data_list,
                all_row_indices,
                all_col_indices,
            ) = self._collect_batch_info(docs)

            # 2) Fetch initial candidates in one batch
            candidates_results = await self._fetch_all_candidates(
                all_entities_to_process,
                all_row_texts,
                all_fuzzies,
                all_qids,
                all_row_indices,
                all_col_indices,
            )

            # 3) Process each row (enhance with LAMAPI features, final ranking, DB update)
            await self._process_rows_individually(row_data_list, candidates_results, db)

        except Exception:
            self.mongo_wrapper.log_to_db(
                "ERROR", "Error processing batch of rows", traceback.format_exc()
            )

    # --------------------------------------------------------------------------
    # 1) GATHER BATCH INFO
    # --------------------------------------------------------------------------
    def _collect_batch_info(self, docs):
        """
        Collects and returns all the lists needed for the candidate-fetch step
        plus a row_data_list for further processing.
        """
        all_entities_to_process = []
        all_row_texts = []
        all_fuzzies = []
        all_qids = []
        all_row_indices = []
        all_col_indices = []
        row_data_list = []

        for doc in docs:
            row = doc["data"]
            ne_columns = doc["classified_columns"]["NE"]
            context_columns = doc.get("context_columns", [])
            correct_qids = doc.get("correct_qids", {})
            row_index = doc.get("row_id", None)

            # Build a text from the "context_columns"
            raw_context_text = " ".join(
                str(row[int(c)])
                for c in sorted(context_columns, key=lambda col: str(row[int(col)]))
            )
            normalized_row_text = raw_context_text.lower()
            normalized_row_text = " ".join(normalized_row_text.split())
            row_hash = hashlib.sha256(normalized_row_text.encode()).hexdigest()

            # Collect row-level info
            row_data_list.append(
                (
                    doc["_id"],
                    row,
                    ne_columns,
                    context_columns,
                    correct_qids,
                    row_index,
                    raw_context_text,
                    row_hash,
                    doc["classified_columns"].get("LIT", {}),  # Add LIT column info
                )
            )

            # Collect all named-entity columns for candidate fetch
            for c, ner_type in ne_columns.items():
                c = str(c)
                if int(c) < len(row):
                    ne_value = row[int(c)]
                    if ne_value and pd.notna(ne_value):
                        ne_value = str(ne_value).strip().replace("_", " ").lower()
                        correct_qid = correct_qids.get(f"{row_index}-{c}", None)

                        all_entities_to_process.append(ne_value)
                        all_row_texts.append(raw_context_text)
                        all_fuzzies.append(False)
                        all_qids.append(correct_qid)
                        all_row_indices.append(row_index)
                        all_col_indices.append(c)

        return (
            all_entities_to_process,
            all_row_texts,
            all_fuzzies,
            all_qids,
            row_data_list,
            all_row_indices,
            all_col_indices,
        )

    # --------------------------------------------------------------------------
    # 2) FETCH INITIAL CANDIDATES + FUZZY RETRY
    # --------------------------------------------------------------------------
    async def _fetch_all_candidates(
        self,
        all_entities_to_process,
        all_row_texts,
        all_fuzzies,
        all_qids,
        all_row_indices,
        all_col_indices,
    ):
        """
        Performs the initial batch fetch of candidates, then does fuzzy retry
        for any entity that returned <= 1 candidate.
        """
        # 1) Initial fetch
        candidates_results = await self.candidate_fetcher.fetch_candidates_batch(
            all_entities_to_process, all_row_texts, all_fuzzies, all_qids
        )

        # 2) Fuzzy retry if needed
        entities_to_retry = []
        row_texts_retry = []
        fuzzies_retry = []
        qids_retry = []

        for ne_value, r_index, c_index in zip(
            all_entities_to_process, all_row_indices, all_col_indices
        ):
            candidates = candidates_results.get(ne_value, [])
            if len(candidates) <= 1:
                entities_to_retry.append(ne_value)
                idx = all_entities_to_process.index(ne_value)
                row_texts_retry.append(all_row_texts[idx])
                fuzzies_retry.append(True)
                qids_retry.append(all_qids[idx])

        if entities_to_retry:
            retry_results = await self.candidate_fetcher.fetch_candidates_batch(
                entities_to_retry, row_texts_retry, fuzzies_retry, qids_retry
            )
            for ne_value in entities_to_retry:
                candidates_results[ne_value] = retry_results.get(ne_value, [])

        return candidates_results

    # --------------------------------------------------------------------------
    # 3) PROCESS EACH ROW INDIVIDUALLY
    # --------------------------------------------------------------------------
    async def _process_rows_individually(self, row_data_list, candidates_results, db):
        """Process rows and store both linked entities and
        training candidates in input_collection"""
        for (
            doc_id,
            row,
            ne_columns,
            context_columns,
            correct_qids,
            row_index,
            raw_context_text,
            row_hash,
            lit_columns,
        ) in row_data_list:
            # Gather all QIDs in this row
            self._collect_row_qids(ne_columns, row, candidates_results)

            for c, ner_type in ne_columns.items():
                c = str(c)
                if int(c) < len(row):
                    ne_value = row[int(c)]
                    if ne_value and pd.notna(ne_value):
                        ne_value = str(ne_value).strip().replace("_", " ").lower()
                        candidates = candidates_results.get(ne_value, [])
                        row_tokens = set(tokenize_text(ne_value))
                        candidates_results[ne_value] = self.feature.process_candidates(
                            candidates, ne_value, row_tokens
                        )

            # Fetch additional features if LAMAPI fetchers are available
            if self.object_fetcher and self.literal_fetcher:
                await self._enhance_with_lamapi_features(
                    ne_columns, lit_columns, row, candidates_results, raw_context_text
                )

            # Build results
            training_candidates = self._build_linked_entities_and_training(
                ne_columns, row, correct_qids, row_index, candidates_results
            )

            # Store everything in the input collection
            db[self.input_collection].update_one(
                {"_id": doc_id},
                {
                    "$set": {
                        "candidates": training_candidates,  # Store training candidates here
                        "status": "DONE",
                        "rank_status": "TODO",
                        "rerank_status": "TODO",
                    }
                },
            )

    def _collect_row_qids(self, ne_columns, row, candidates_results):
        """
        Collects the QIDs for all entities in a single row.
        """
        row_qids = []
        for c, ner_type in ne_columns.items():
            if int(c) < len(row):
                ne_value = row[int(c)]
                if ne_value and pd.notna(ne_value):
                    ne_value = str(ne_value).strip().replace("_", " ").lower()
                    candidates = candidates_results.get(ne_value, [])
                    for cand in candidates:
                        if cand["id"]:
                            row_qids.append(cand["id"])
        return list(set(q for q in row_qids if q))

    async def _enhance_with_lamapi_features(
        self, ne_columns, lit_columns, row, candidates_results, raw_context_text
    ):
        """Enhance candidate features with LAMAPI data: objects and literals."""
        if not self.object_fetcher or not self.literal_fetcher:
            return

        # Collect all candidates from NE cells in this row
        all_candidates_by_col = {}
        all_entity_ids = set()

        for c, ner_type in ne_columns.items():
            c_int = int(c)
            if c_int < len(row):
                ne_value = row[c_int]
                if ne_value and pd.notna(ne_value):
                    ne_value = str(ne_value).strip().replace("_", " ").lower()
                    candidates = candidates_results.get(ne_value, [])

                    if candidates:
                        all_candidates_by_col[c] = candidates
                        for cand in candidates:
                            if cand.get("id"):
                                all_entity_ids.add(cand["id"])
                                # Add structures to track relationships
                                cand.setdefault("matches", defaultdict(list))
                                cand.setdefault("predicates", defaultdict(dict))

        if not all_entity_ids:
            return

        # Fetch objects and literals data
        objects_data = await self.object_fetcher.fetch_objects(list(all_entity_ids))
        literals_data = await self.literal_fetcher.fetch_literals(list(all_entity_ids))

        if not objects_data and not literals_data:
            return

        # Process NE-NE relationships
        self.feature.compute_entity_entity_relationships(all_candidates_by_col, objects_data)

        # Process NE-LIT relationships
        self.feature.compute_entity_literal_relationships(
            all_candidates_by_col, lit_columns, row, literals_data
        )

    def _build_linked_entities_and_training(
        self, ne_columns, row, correct_qids, row_index, candidates_results
    ):
        """
        For each NE column in the row:
          - Insert column_NERtype features
          - Rank candidates
          - Insert correct candidate if missing
          - Return final top K + training slice
        """
        training_candidates_by_ne_column = {}

        for c, ner_type in ne_columns.items():
            c = str(c)
            if int(c) < len(row):
                ne_value = row[int(c)]
                if ne_value and pd.notna(ne_value):
                    ne_value = str(ne_value).strip().replace("_", " ").lower()
                    candidates = candidates_results.get(ne_value, [])

                    # Rank
                    max_training_candidates = len(candidates)
                    ranked_candidates = candidates
                    ranked_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)

                    # If correct QID is missing in top training slice, insert it
                    correct_qid = correct_qids.get(f"{row_index}-{c}", None)
                    if correct_qid and correct_qid not in [
                        rc["id"] for rc in ranked_candidates[:max_training_candidates]
                    ]:
                        correct_candidate = next(
                            (x for x in ranked_candidates if x["id"] == correct_qid), None
                        )
                        if correct_candidate:
                            top_slice = ranked_candidates[: max_training_candidates - 1]
                            top_slice.append(correct_candidate)
                            ranked_candidates = top_slice
                            ranked_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)

                    # Slice final results
                    training_candidates = ranked_candidates[:max_training_candidates]
                    training_candidates_by_ne_column[c] = training_candidates

        return training_candidates_by_ne_column
