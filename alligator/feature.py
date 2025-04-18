from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from pymongo.collection import Collection

from alligator.database import DatabaseAccessMixin
from alligator.typing import Candidate, LiteralsData, ObjectsData
from alligator.utils import ColumnHelper, ngrams, tokenize_text

DEFAULT_FEATURES = [
    "ambiguity_mention",
    "ncorrects_tokens",
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
        top_n_for_type_freq: int = 5,
        features: Optional[List[str]] = None,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.table_name = table_name
        self.top_n_for_type_freq = top_n_for_type_freq
        self.selected_features = features or DEFAULT_FEATURES
        self._db_name = kwargs.pop("db_name", "alligator_db")
        self._mongo_uri = kwargs.pop("mongo_uri", "mongodb://gator-mongodb:27017/")
        self.input_collection = kwargs.get("input_collection", "input_data")

    def map_kind_to_numeric(self, kind: str) -> int:
        mapping: Dict[str, int] = {
            "entity": 1,
            "type": 2,
            "disambiguation": 3,
            "predicate": 4,
        }
        return mapping.get(kind, 1)

    def calculate_token_overlap(self, tokens_a: Set[str], tokens_b: Set[str]) -> float:
        intersection: Set[str] = tokens_a & tokens_b
        union: Set[str] = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0.0

    def calculate_ngram_similarity(self, a: str, b: str, n: int = 3) -> float:
        a_ngrams: List[str] = ngrams(a, n)
        b_ngrams: List[str] = ngrams(b, n)
        intersection: int = len(set(a_ngrams) & set(b_ngrams))
        union: int = len(set(a_ngrams) | set(b_ngrams))
        return intersection / union if union > 0 else 0.0

    def process_candidates(
        self, candidates: List[Candidate], entity_name: Optional[str], row_tokens: Set[str]
    ) -> None:
        """
        Process candidate records to calculate a set of features for each candidate.
        """
        # Use a safe version of entity_name for computations.
        safe_entity_name: str = entity_name if entity_name is not None else ""

        for candidate in candidates:
            # Retrieve original values, which might be None.
            candidate_name: Optional[str] = candidate.name
            candidate_description: Optional[str] = candidate.description

            # Create safe versions for computation.
            safe_candidate_name: str = candidate_name if candidate_name is not None else ""
            safe_candidate_description: str = (
                candidate_description if candidate_description is not None else ""
            )

            desc: float = 0.0
            descNgram: float = 0.0
            if safe_candidate_description:
                desc = self.calculate_token_overlap(
                    row_tokens, tokenize_text(safe_candidate_description)
                )
                descNgram = self.calculate_ngram_similarity(
                    safe_entity_name, safe_candidate_description
                )

            # Initialize all default features with default values
            candidate_features: Dict[str, Any] = candidate.features
            features: Dict[str, Any] = {feature: 0.0 for feature in DEFAULT_FEATURES}

            # Now set the calculated values in the same order as DEFAULT_FEATURES
            features.update(
                {
                    "ambiguity_mention": candidate_features.get("ambiguity_mention", 0.0),
                    "ncorrects_tokens": candidate_features.get("ncorrects_tokens", 0.0),
                    "ntoken_mention": candidate_features.get(
                        "ntoken_mention", len(safe_entity_name.split())
                    ),
                    "ntoken_entity": candidate_features.get(
                        "ntoken_entity", len(safe_candidate_name.split())
                    ),
                    "length_mention": len(safe_entity_name),
                    "length_entity": len(safe_candidate_name),
                    "popularity": candidate_features.get("popularity", 0.0),
                    "pos_score": candidate_features.get("pos_score", 0.0),
                    "es_score": candidate_features.get("es_score", 0.0),
                    "ed_score": candidate_features.get("ed_score", 0.0),
                    "jaccard_score": candidate_features.get("jaccard_score", 0.0),
                    "jaccardNgram_score": candidate_features.get("jaccardNgram_score", 0.0),
                    "p_subj_ne": candidate_features.get("p_subj_ne", 0.0),
                    "p_subj_lit_datatype": candidate_features.get("p_subj_lit_datatype", 0.0),
                    "p_subj_lit_all_datatype": candidate_features.get(
                        "p_subj_lit_all_datatype", 0.0
                    ),
                    "p_subj_lit_row": candidate_features.get("p_subj_lit_row", 0.0),
                    "p_obj_ne": candidate_features.get("p_obj_ne", 0.0),
                    "desc": desc,
                    "descNgram": descNgram,
                    "cta_t1": candidate_features.get("cta_t1", 0.0),
                    "cta_t2": candidate_features.get("cta_t2", 0.0),
                    "cta_t3": candidate_features.get("cta_t3", 0.0),
                    "cta_t4": candidate_features.get("cta_t4", 0.0),
                    "cta_t5": candidate_features.get("cta_t5", 0.0),
                    "cpa_t1": candidate_features.get("cpa_t1", 0.0),
                    "cpa_t2": candidate_features.get("cpa_t2", 0.0),
                    "cpa_t3": candidate_features.get("cpa_t3", 0.0),
                    "cpa_t4": candidate_features.get("cpa_t4", 0.0),
                    "cpa_t5": candidate_features.get("cpa_t5", 0.0),
                }
            )

            # Preserve the original candidate values, even if they are None
            candidate.features = features

    def compute_global_type_frequencies(
        self,
        docs_to_process: Optional[float] = None,
        random_sample: bool = False,
        doc_range: Optional[tuple] = None,
    ) -> Dict[Any, Counter]:
        """Compute type frequencies across candidate documents.

        Args:
            docs_to_process: Percentage of documents to process. If None, processes all documents.
                Use this to limit computation time for very large datasets.
            random_sample: If True and docs_to_process is specified, samples documents randomly.
            doc_range: Optional tuple (start, end) to process a specific document range.
                Takes precedence over random_sample.

        Returns:
            Dictionary mapping column indexes to type frequency counters.
        """
        col: Collection = self.get_db()[self.input_collection]
        type_freq_by_column = defaultdict(Counter)
        rows_count_by_column = Counter()

        # Base query to find documents with candidates
        match_query = {
            "dataset_name": self.dataset_name,
            "table_name": self.table_name,
            "status": "DONE",
            "rank_status": "DONE",
            "candidates": {"$exists": True},
        }
        projection = {"candidates": 1}

        # Choose sampling strategy
        if doc_range:
            start, end = doc_range
            print(f"Processing documents in range {start} to {end}")
            cursor = col.find(match_query, projection).skip(start).limit(end - start)
        elif docs_to_process and random_sample:
            # Use aggregation pipeline with $sample for random sampling
            total_docs = col.count_documents(match_query)
            max_docs = max(1, int(total_docs * docs_to_process))
            print(f"Computing type-frequency features by randomly sampling {max_docs} documents")
            pipeline = [
                {"$match": match_query},
                {"$sample": {"size": max_docs}},
                {"$project": projection},
            ]
            cursor = col.aggregate(pipeline)
        else:
            # Simple limit if specified
            cursor = col.find(match_query, projection)
            if docs_to_process:
                total_docs = col.count_documents(match_query)
                max_docs = max(1, int(total_docs * docs_to_process))
                print(
                    f"Computing type-frequency features by processing first {max_docs} documents"
                )
                cursor = cursor.limit(max_docs)
            else:
                print("Computing type-frequency features by processing all matching documents")

        # Process documents to calculate type frequencies
        doc_count = 0
        for doc in cursor:
            doc_count += 1
            candidates_by_column: Dict[Any, List[Dict[str, Any]]] = doc["candidates"]

            for col_idx, candidates in candidates_by_column.items():
                top_candidates = candidates[: self.top_n_for_type_freq]
                row_qids = set()

                for cand in top_candidates:
                    for t_dict in cand.get("types", []):
                        qid = t_dict.get("id")
                        if qid:
                            row_qids.add(qid)

                for qid in row_qids:
                    type_freq_by_column[col_idx][qid] += 1

                rows_count_by_column[col_idx] += 1

        # Normalize frequencies
        for col_idx, freq_counter in type_freq_by_column.items():
            row_count: int = rows_count_by_column[col_idx]
            if row_count == 0:
                continue
            # Convert each type's raw count to a ratio in [0..1]
            for qid in freq_counter:
                freq_counter[qid] = freq_counter[qid] / row_count

        print(f"Computed type frequencies from {doc_count} documents")
        return type_freq_by_column

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
                for subj_candidate in subj_candidates:
                    subj_id = subj_candidate.id
                    if not subj_id or subj_id not in objects_data:
                        continue

                    # Get the objects related to this subject
                    subj_objects = objects_data[subj_id].get("objects", {})
                    objects_set = set(subj_objects.keys())

                    # Get object candidates' IDs in this column
                    obj_candidate_ids = {oc.id for oc in obj_candidates if oc.id}
                    objects_intersection = objects_set.intersection(obj_candidate_ids)

                    # Skip if no intersection
                    if not objects_intersection:
                        continue

                    # Calculate maximum object score for this subject
                    obj_score_max = 0
                    object_rel_score_buffer = {}

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
                        object_rel_score_buffer[obj_id] = object_rel_score_buffer.get(obj_id, 0)

                        # Calculate reverse score from subject to object
                        subj_string_features = []
                        for feature_name in ["ed_score", "jaccard_score", "jaccardNgram_score"]:
                            if feature_name in subj_candidate.features:
                                subj_string_features.append(subj_candidate.features[feature_name])

                        if subj_string_features:
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
                    if obj_score_max > 0:
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
        row_text_all = " ".join(str(v) for v in row if pd.notna(v)).lower()
        row_text_lit = " ".join(
            str(row[int(c)]) for c in lit_columns if int(c) < len(row) and pd.notna(row[int(c)])
        ).lower()

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
                        lit_values.extend(
                            str(val).lower() for val in subj_literals[datatype][predicate]
                        )

                lit_string = " ".join(lit_values)

                # Calculate token-based similarity

                lit_tokens = set(tokenize_text(lit_string))
                row_all_tokens = set(tokenize_text(row_text_all))
                row_lit_tokens = set(tokenize_text(row_text_lit))

                p_subj_lit_all_datatype = (
                    len(lit_tokens & row_lit_tokens) / len(lit_tokens | row_lit_tokens)
                    if lit_tokens and row_lit_tokens
                    else 0
                )
                p_subj_lit_row = (
                    len(lit_tokens & row_all_tokens) / len(lit_tokens | row_all_tokens)
                    if lit_tokens and row_all_tokens
                    else 0
                )

                subj_candidate.features["p_subj_lit_all_datatype"] = p_subj_lit_all_datatype
                subj_candidate.features["p_subj_lit_row"] = p_subj_lit_row

                # Process each literal column
                for lit_col, lit_type in lit_columns.items():
                    normalized_lit_col = ColumnHelper.normalize(lit_col)
                    if not ColumnHelper.is_valid_index(normalized_lit_col, len(row)):
                        continue

                    lit_value = row[ColumnHelper.to_int(normalized_lit_col)]
                    if not lit_value or pd.isna(lit_value):
                        continue

                    lit_value = str(lit_value).lower()
                    lit_datatype = lit_type.upper()  # Normalize datatype

                    # Find matching datatype in literals
                    normalized_datatype = lit_datatype.lower()
                    if normalized_datatype not in subj_literals:
                        continue

                    # Calculate maximum similarity for this literal column
                    max_score = 0.0
                    for predicate in subj_literals[normalized_datatype]:
                        for kg_value in subj_literals[normalized_datatype][predicate]:
                            kg_value = str(kg_value).lower()

                            # Calculate similarity based on datatype
                            p_subj_lit = 0.0
                            if lit_datatype == "NUMBER":
                                # Simple numeric similarity (could be improved)
                                try:
                                    num1 = float(lit_value)
                                    num2 = float(kg_value)
                                    if num1 == num2:
                                        p_subj_lit = 1.0
                                    else:
                                        max_val = max(abs(num1), abs(num2))
                                        diff = abs(num1 - num2)
                                        if max_val > 0:
                                            p_subj_lit = max(0, 1 - (diff / max_val))
                                except (ValueError, TypeError):
                                    pass
                            elif lit_datatype == "DATETIME":
                                # Simple string comparison for dates (could be improved)
                                if lit_value == kg_value:
                                    p_subj_lit = 1.0
                                else:
                                    # Simple substring match
                                    p_subj_lit = (
                                        0.5
                                        if (lit_value in kg_value or kg_value in lit_value)
                                        else 0
                                    )
                            else:  # STRING
                                # Use token-based similarity
                                lit_tokens = set(tokenize_text(lit_value))
                                kg_tokens = set(tokenize_text(kg_value))
                                if lit_tokens and kg_tokens:
                                    p_subj_lit = len(lit_tokens & kg_tokens) / len(
                                        lit_tokens | kg_tokens
                                    )

                            if p_subj_lit > 0:
                                # Record match
                                subj_candidate.matches[normalized_lit_col].append(
                                    {"p": predicate, "o": kg_value, "s": p_subj_lit}
                                )

                                # Update maximum score
                                max_score = max(max_score, p_subj_lit)

                                # Update predicates
                                if predicate not in subj_candidate.predicates[normalized_lit_col]:
                                    subj_candidate.predicates[normalized_lit_col][predicate] = 0
                                subj_candidate.predicates[normalized_lit_col][predicate] = max(
                                    subj_candidate.predicates[normalized_lit_col][predicate],
                                    p_subj_lit,
                                )

                    # Normalize and update feature
                    if max_score > 0:
                        subj_candidate.features["p_subj_lit_datatype"] += max_score / n_lit_cols
