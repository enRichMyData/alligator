from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from alligator.database import DatabaseAccessMixin
from alligator.typing import Candidate, LiteralsData, ObjectsData
from alligator.utils import (
    ColumnHelper,
    clean_str,
    compute_similarity_between_dates,
    compute_similarity_between_string,
    compute_similarity_between_string_token_based,
    compute_similarty_between_numbers,
)

DEFAULT_FEATURES = [
    "ambiguity_mention",
    "corrects_tokens",
    "ntoken_mention",
    "ntoken_entity",
    "length_mention",
    "length_entity",
    "popularity",
    "pos_score",
    "es_score",
    "ed_score",
    "jaccard_score",
    "jaccardNgram_score",
    "p_subj_ne",
    "p_subj_lit_datatype",
    "p_subj_lit_all_datatype",
    "p_subj_lit_row",
    "p_obj_ne",
    "desc",
    "descNgram",
    "cta_t1",
    "cta_t2",
    "cta_t3",
    "cta_t4",
    "cta_t5",
    "cpa_t1",
    "cpa_t2",
    "cpa_t3",
    "cpa_t4",
    "cpa_t5",
]


class Feature(DatabaseAccessMixin):
    def __init__(
        self,
        dataset_name: str,
        table_name: str,
        top_n_cta_cpa_freq: int = 3,
        features: Optional[List[str]] = None,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.table_name = table_name
        self.top_n_cta_cpa_freq = top_n_cta_cpa_freq
        self.selected_features = features or DEFAULT_FEATURES
        self._db_name = kwargs.pop("db_name", "alligator_db")
        self._mongo_uri = kwargs.pop("mongo_uri", "mongodb://gator-mongodb:27017/")
        self.input_collection = kwargs.get("input_collection", "input_data")
        self.candidate_collection = kwargs.get("candidate_collection", "candidates")
        self._predicate_frequencies = None

    def map_kind_to_numeric(self, kind: str) -> int:
        mapping: Dict[str, int] = {
            "entity": 1,
            "type": 2,
            "disambiguation": 3,
            "predicate": 4,
        }
        return mapping.get(kind, 1)

    def process_candidates(self, candidates: List[Candidate], row: Optional[str]) -> None:
        """
        Process candidate records to calculate a set of features for each candidate.
        """
        # Use a safe version of entity_name for computations.
        safe_row = row if row is not None else ""
        safe_row = clean_str(safe_row)

        for candidate in candidates:
            # Retrieve original values, which might be None.
            candidate_name: Optional[str] = candidate.name
            candidate_description: Optional[str] = candidate.description

            # Create safe versions for computation.
            safe_candidate_name: str = candidate_name if candidate_name is not None else ""
            safe_candidate_name = clean_str(safe_candidate_name)
            safe_candidate_description: str = (
                candidate_description if candidate_description is not None else ""
            )
            safe_candidate_description = clean_str(safe_candidate_description)

            desc: float = 0.0
            descNgram: float = 0.0
            if safe_candidate_description:
                desc = compute_similarity_between_string(
                    safe_candidate_description, row, ngram=None
                )
                descNgram = compute_similarity_between_string(
                    safe_candidate_description, row, ngram=3
                )

            # Initialize all default features with default values
            candidate_features: Dict[str, Any] = candidate.features
            features: Dict[str, Any] = {
                feature: candidate_features.get(feature, 0.0) for feature in DEFAULT_FEATURES
            }
            features["desc"] = desc
            features["descNgram"] = descNgram

            # Preserve the original candidate values, even if they are None
            candidate.features = features

    def compute_global_frequencies(
        self, docs_to_process: float = 1.0, random_sample: bool = False
    ) -> Tuple[Dict[Any, Counter], Dict[Any, Counter], Dict[Any, Dict[Any, Counter]]]:
        """
        Compute global type frequencies (CTA) and predicate frequencies (CPA) across all columns.

        Args:
            docs_to_process: Fraction of documents to use (0.0-1.0)
            random_sample: Whether to randomly sample documents

        Returns:
            Tuple of (type_frequencies, predicate_frequencies) where each is a dict
            mapping column indices to Counter objects with frequency data
        """
        db = self.get_db()
        input_collection = db[self.input_collection]

        # Build the base query
        query = {
            "dataset_name": self.dataset_name,
            "table_name": self.table_name,
            "status": "DONE",
            "rank_status": "DONE",
        }

        # Determine the number of documents to get
        total_docs_matching_query = input_collection.count_documents(query)
        if docs_to_process is None or docs_to_process >= 1.0:
            docs_to_get = total_docs_matching_query
        else:
            docs_to_get = max(1, int(total_docs_matching_query * docs_to_process))

        if docs_to_get == 0:
            print("No documents match the criteria for computing global frequencies.")
            return defaultdict(Counter), defaultdict(Counter)

        # Base pipeline stages
        pipeline: List[Dict[str, Any]] = [{"$match": query}]

        if random_sample:
            print(
                f"Computing type-frequency features by randomly sampling {docs_to_get} documents"
            )
            pipeline.append({"$sample": {"size": docs_to_get}})
        else:
            print(f"Computing type-frequency features by processing first {docs_to_get} documents")
            pipeline.append({"$limit": docs_to_get})

        # Add $lookup to join with candidates
        pipeline.extend(
            [
                {
                    "$lookup": {
                        "from": self.candidate_collection,
                        "let": {"doc_id": "$_id", "row_id_str": {"$toString": "$row_id"}},
                        "pipeline": [
                            {
                                "$match": {
                                    "$expr": {
                                        "$and": [
                                            {"$eq": ["$owner_id", "$$doc_id"]},
                                            {"$eq": ["$row_id", "$$row_id_str"]},
                                        ]
                                    }
                                }
                            },
                            {"$project": {"_id": 0, "candidates": 1, "col_id": 1}},
                        ],
                        "as": "candidate_records_array",
                    }
                },
                {
                    "$addFields": {
                        "candidates_by_column": {
                            "$arrayToObject": {
                                "$map": {
                                    "input": "$candidate_records_array",
                                    "as": "cand_rec",
                                    "in": {"k": "$$cand_rec.col_id", "v": "$$cand_rec.candidates"},
                                }
                            }
                        }
                    }
                },
                {"$project": {"classified_columns": 1, "candidates_by_column": 1, "_id": 0}},
            ]
        )

        cursor = input_collection.aggregate(pipeline)

        # Initialize counters
        n_docs = 0
        predicate_pair_frequencies = {}
        type_frequencies = defaultdict(Counter)
        predicate_frequencies = defaultdict(Counter)

        for doc in cursor:
            n_docs += 1
            candidates_by_column = doc.get("candidates_by_column", {})
            ne_cols = doc.get("classified_columns", {}).get("NE", {})

            for col_idx in ne_cols:
                candidates = candidates_by_column.get(col_idx, [])
                if not candidates:
                    continue

                if col_idx not in predicate_pair_frequencies:
                    predicate_pair_frequencies[col_idx] = {}

                seen_types = set()
                seen_predicates = set()

                for candidate in candidates[: self.top_n_cta_cpa_freq]:
                    for type_obj in candidate.get("types", []):
                        type_id = type_obj.get("id")
                        if type_id and type_id not in seen_types:
                            type_frequencies[col_idx][type_id] += 1
                            seen_types.add(type_id)

                    predicates = candidate.get("predicates", {})
                    for rel_col_idx, rel_predicates in predicates.items():
                        if rel_col_idx not in predicate_pair_frequencies[col_idx]:
                            predicate_pair_frequencies[col_idx][rel_col_idx] = Counter()
                        for pred_id, pred_value in rel_predicates.items():
                            if pred_id and pred_id not in seen_predicates:
                                predicate_frequencies[col_idx][pred_id] += pred_value
                                predicate_pair_frequencies[col_idx][rel_col_idx][
                                    pred_id
                                ] += pred_value
                                seen_predicates.add(pred_id)

        if n_docs == 0 and total_docs_matching_query > 0 and docs_to_get > 0:
            print(
                f"Warning: {docs_to_get} documents were selected for frequency computation, "
                "but the aggregation pipeline (possibly $lookup) returned no combined documents. "
                "Check if candidates exist for these documents or "
                "if the $lookup conditions are too restrictive."
            )

        if n_docs > 0:
            for col_idx in type_frequencies:
                for type_id in type_frequencies[col_idx]:
                    type_frequencies[col_idx][type_id] = (
                        type_frequencies[col_idx][type_id] / n_docs
                    )

            for col_idx in predicate_frequencies:
                for pred_id in predicate_frequencies[col_idx]:
                    predicate_frequencies[col_idx][pred_id] = (
                        predicate_frequencies[col_idx][pred_id] / n_docs
                    )

        return type_frequencies, predicate_frequencies, predicate_pair_frequencies

    def compute_entity_entity_relationships(
        self,
        all_candidates_by_col: Dict[str, List[Candidate]],
        objects_data: Dict[str, ObjectsData],
    ) -> None:
        """
        Compute relationships between named entities.
        """
        if not all_candidates_by_col or len(all_candidates_by_col) <= 1:
            return

        # Count the number of NE columns for normalization
        n_ne_cols = len(all_candidates_by_col)

        # Compare each NE column with every other NE column
        for subj_col, subj_candidates in all_candidates_by_col.items():
            for obj_col, obj_candidates in all_candidates_by_col.items():
                if subj_col == obj_col:
                    continue

                # Process each subject candidate
                object_rel_score_buffer = {}
                obj_candidate_ids = {oc.id for oc in obj_candidates if oc.id}

                for subj_candidate in subj_candidates:
                    subj_id = subj_candidate.id
                    if not subj_id or subj_id not in objects_data:
                        continue

                    # Get the objects related to this subject
                    subj_objects = objects_data[subj_id].get("objects", {})
                    if not subj_objects:
                        continue
                    subj_objects_set = set(subj_objects.keys())

                    # Get object candidates' IDs in this column
                    objects_intersection = subj_objects_set.intersection(obj_candidate_ids)

                    # Skip if no intersection
                    if not objects_intersection:
                        continue

                    # Calculate maximum object score for this subject
                    obj_score_max = 0.0

                    for obj_candidate in obj_candidates:
                        obj_id = obj_candidate.id
                        if not obj_id or obj_id not in objects_intersection:
                            continue

                        # Calculate similarity score (average of string similarity features)
                        string_features = []
                        for feature_name in ["ed_score", "jaccard_score", "jaccardNgram_score"]:
                            if feature_name in obj_candidate.features:
                                string_features.append(obj_candidate.features[feature_name])

                        if not string_features:
                            continue

                        p_subj_ne = sum(string_features) / len(string_features)
                        obj_score_max = max(obj_score_max, p_subj_ne)

                        # Track the best score for each object
                        if obj_id not in object_rel_score_buffer:
                            object_rel_score_buffer[obj_id] = 0

                        # Calculate reverse score from subject to object
                        subj_string_features = []
                        for feature_name in ["ed_score", "jaccard_score", "jaccardNgram_score"]:
                            if feature_name in subj_candidate.features:
                                subj_string_features.append(subj_candidate.features[feature_name])

                        score_rel = sum(subj_string_features) / len(subj_string_features)
                        object_rel_score_buffer[obj_id] = max(
                            object_rel_score_buffer[obj_id], score_rel
                        )

                        # Record predicates connecting subject to object
                        for predicate in subj_objects.get(obj_id, []):
                            subj_candidate.matches[obj_col].append(
                                {"p": predicate, "o": obj_id, "s": p_subj_ne}
                            )
                            subj_candidate.predicates[obj_col][predicate] = p_subj_ne

                    # Normalize and update subject's feature
                    subj_candidate.features["p_subj_ne"] += obj_score_max / n_ne_cols

                # Update object candidates' features
                for obj_candidate in obj_candidates:
                    obj_id = obj_candidate.id
                    if obj_id in object_rel_score_buffer:
                        obj_candidate.features["p_obj_ne"] += (
                            object_rel_score_buffer[obj_id] / n_ne_cols
                        )

    def compute_entity_literal_relationships(
        self,
        all_candidates_by_col: Dict[str, List[Candidate]],
        lit_columns: Dict[str, str],
        row: List[Any],
        literals_data: Dict[str, LiteralsData],
    ) -> None:
        """
        Compute relationships between named entities and literals.
        """
        if not all_candidates_by_col or not lit_columns:
            return

        # Count the number of LIT columns for normalization
        n_lit_cols = len(lit_columns)
        if n_lit_cols == 0:
            return

        # Get row text tokens
        row_text_all = clean_str(" ".join(str(v) for v in row))
        row_text_lit = clean_str(" ".join(str(row[int(c)]) for c in lit_columns))

        # Process each subject (NE) candidate
        for subj_col, subj_candidates in all_candidates_by_col.items():
            for subj_candidate in subj_candidates:
                subj_id = subj_candidate.id
                if not subj_id or subj_id not in literals_data:
                    continue

                # Get literals for this subject
                subj_literals = literals_data[subj_id].get("literals", {})
                if not subj_literals:
                    continue

                # Calculate row-wide literal features
                lit_values = []
                for datatype in subj_literals:
                    for predicate in subj_literals[datatype]:
                        for value in subj_literals[datatype][predicate]:
                            if value.startswith("+") and value[1:].isdigit():
                                value = value[1:]
                            lit_values.append(str(value).lower())
                lit_string = " ".join(lit_values)

                subj_candidate.features[
                    "p_subj_lit_all_datatype"
                ] = compute_similarity_between_string_token_based(lit_string, row_text_lit)
                subj_candidate.features[
                    "p_subj_lit_row"
                ] = compute_similarity_between_string_token_based(lit_string, row_text_all)

                # Process each literal column
                for lit_col, lit_type in lit_columns.items():
                    normalized_lit_col = ColumnHelper.normalize(lit_col)
                    if not ColumnHelper.is_valid_index(normalized_lit_col, len(row)):
                        continue

                    lit_value = row[ColumnHelper.to_int(normalized_lit_col)]
                    if not lit_value or pd.isna(lit_value):
                        continue

                    lit_value = str(lit_value).lower()
                    normalized_datatype = lit_type.upper()
                    if normalized_datatype not in subj_literals:
                        continue

                    # Calculate maximum similarity for this literal column
                    max_score = 0.0
                    for predicate in subj_literals[normalized_datatype]:
                        for kg_value in subj_literals[normalized_datatype][predicate]:
                            kg_value = str(kg_value).lower()

                            # Calculate similarity based on datatype
                            p_subj_lit = 0.0
                            if lit_type == "NUMBER":
                                p_subj_lit = compute_similarty_between_numbers(lit_value, kg_value)
                            elif normalized_datatype == "DATETIME":
                                p_subj_lit = compute_similarity_between_dates(lit_value, kg_value)
                            elif normalized_datatype == "STRING":
                                p_subj_lit = compute_similarity_between_string(lit_value, kg_value)
                            if p_subj_lit > 0:
                                subj_candidate.matches[normalized_lit_col].append(
                                    {"p": predicate, "o": kg_value, "s": p_subj_lit}
                                )
                                max_score = max(max_score, p_subj_lit)

                                # Update predicates
                                if predicate not in subj_candidate.predicates[normalized_lit_col]:
                                    subj_candidate.predicates[normalized_lit_col][predicate] = 0
                                subj_candidate.predicates[normalized_lit_col][predicate] = max(
                                    subj_candidate.predicates[normalized_lit_col][predicate],
                                    p_subj_lit,
                                )

                    # Normalize and update feature
                    subj_candidate.features["p_subj_lit_datatype"] += max_score / n_lit_cols
