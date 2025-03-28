import hashlib
import traceback
import asyncio
from collections import defaultdict

import pandas as pd

from alligator.fetchers import CandidateFetcher, ObjectFetcher, LiteralFetcher
from alligator.mongo import MongoWrapper
from alligator.utils import tokenize_text


class RowBatchProcessor:
    """
    Extracted logic for process_rows_batch (and associated scoring helpers).
    Takes the Alligator instance so we can reference .mongo_wrapper, .feature, etc.
    """

    def __init__(
        self,
        candidate_fetcher: CandidateFetcher,
        object_fetcher: ObjectFetcher = None,
        literal_fetcher: LiteralFetcher = None,
        max_candidates_in_result: int = 5,
        **kwargs,
    ):
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

    def process_rows_batch(self, docs, dataset_name, table_name):
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
            candidates_results = self._fetch_all_candidates(
                all_entities_to_process,
                all_row_texts,
                all_fuzzies,
                all_qids,
                all_row_indices,
                all_col_indices,
            )

            # 3) Process each row (enhance with LAMAPI features, final ranking, DB update)
            self._process_rows_individually(row_data_list, candidates_results, db)

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
    def _fetch_all_candidates(
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
        candidates_results = self.candidate_fetcher.fetch_candidates_batch(
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
            retry_results = self.candidate_fetcher.fetch_candidates_batch(
                entities_to_retry, row_texts_retry, fuzzies_retry, qids_retry
            )
            for ne_value in entities_to_retry:
                candidates_results[ne_value] = retry_results.get(ne_value, [])

        return candidates_results

    # --------------------------------------------------------------------------
    # 3) PROCESS EACH ROW INDIVIDUALLY
    # --------------------------------------------------------------------------
    def _process_rows_individually(self, row_data_list, candidates_results, db):
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
            row_qids = self._collect_row_qids(ne_columns, row, candidates_results)

            # Fetch additional features if LAMAPI fetchers are available
            bow_data = {}
            if self.object_fetcher and self.literal_fetcher:
                self._enhance_with_lamapi_features(
                    ne_columns, lit_columns, row, candidates_results, raw_context_text
                )

            # Build results
            training_candidates = self._build_linked_entities_and_training(
                ne_columns, row, correct_qids, row_index, candidates_results, bow_data
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

    def _enhance_with_lamapi_features(self, ne_columns, lit_columns, row, candidates_results, raw_context_text):
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
        objects_data = asyncio.run(self.object_fetcher.fetch_objects(list(all_entity_ids)))
        literals_data = asyncio.run(self.literal_fetcher.fetch_literals(list(all_entity_ids)))
        
        if not objects_data and not literals_data:
            return
            
        # Process NE-NE relationships
        self._compute_ne_ne_relationships(all_candidates_by_col, objects_data)
        
        # Process NE-LIT relationships
        self._compute_ne_lit_relationships(all_candidates_by_col, lit_columns, row, literals_data, raw_context_text)

    def _compute_ne_ne_relationships(self, all_candidates_by_col, objects_data):
        """Compute relationships between named entities."""
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
                    subj_id = subj_candidate.get("id")
                    if not subj_id or subj_id not in objects_data:
                        continue
                        
                    # Get the objects related to this subject
                    subj_objects = objects_data[subj_id].get("objects", {})
                    objects_set = set(subj_objects.keys())
                    
                    # Get object candidates' IDs in this column
                    obj_candidate_ids = {oc.get("id") for oc in obj_candidates if oc.get("id")}
                    objects_intersection = objects_set.intersection(obj_candidate_ids)
                    
                    # Skip if no intersection
                    if not objects_intersection:
                        continue
                        
                    # Calculate maximum object score for this subject
                    obj_score_max = 0
                    object_rel_score_buffer = {}
                    
                    for obj_candidate in obj_candidates:
                        obj_id = obj_candidate.get("id")
                        if not obj_id or obj_id not in objects_intersection:
                            continue
                            
                        # Calculate similarity score (average of string similarity features)
                        string_features = []
                        for feature_name in ["ed_score", "jaccard_score", "jaccardNgram_score"]:
                            if feature_name in obj_candidate.get("features", {}):
                                string_features.append(obj_candidate["features"][feature_name])
                                
                        if not string_features:
                            continue
                            
                        p_subj_ne = round(sum(string_features) / len(string_features), 3)
                        obj_score_max = max(obj_score_max, p_subj_ne)
                        
                        # Track the best score for each object
                        object_rel_score_buffer[obj_id] = object_rel_score_buffer.get(obj_id, 0)
                        
                        # Calculate reverse score from subject to object
                        subj_string_features = []
                        for feature_name in ["ed_score", "jaccard_score", "jaccardNgram_score"]:
                            if feature_name in subj_candidate.get("features", {}):
                                subj_string_features.append(subj_candidate["features"][feature_name])
                                
                        if subj_string_features:
                            score_rel = round(sum(subj_string_features) / len(subj_string_features), 3)
                            object_rel_score_buffer[obj_id] = max(object_rel_score_buffer[obj_id], score_rel)
                            
                        # Record predicates connecting subject to object
                        for predicate in subj_objects.get(obj_id, []):
                            subj_candidate["matches"][str(obj_col)].append({
                                "p": predicate,
                                "o": obj_id,
                                "s": round(p_subj_ne, 3)
                            })
                            subj_candidate["predicates"][str(obj_col)][predicate] = p_subj_ne
                    
                    # Normalize and update subject's feature
                    if obj_score_max > 0:
                        subj_candidate["features"]["p_subj_ne"] += round(obj_score_max / n_ne_cols, 3)
                    
                    # Update object candidates' features
                    for obj_candidate in obj_candidates:
                        obj_id = obj_candidate.get("id")
                        if obj_id in object_rel_score_buffer:
                            obj_candidate["features"]["p_obj_ne"] += round(object_rel_score_buffer[obj_id] / n_ne_cols, 3)

    def _compute_ne_lit_relationships(self, all_candidates_by_col, lit_columns, row, literals_data, raw_context_text):
        """Compute relationships between named entities and literals."""
        if not all_candidates_by_col or not lit_columns:
            return
            
        # Count the number of LIT columns for normalization
        n_lit_cols = len(lit_columns)
        if n_lit_cols == 0:
            return
            
        # Get row text tokens
        row_text_all = " ".join(str(v) for v in row if pd.notna(v)).lower()
        row_text_lit = " ".join(str(row[int(c)]) for c in lit_columns if int(c) < len(row) and pd.notna(row[int(c)])).lower()
        
        # Process each subject (NE) candidate
        for subj_col, subj_candidates in all_candidates_by_col.items():
            for subj_candidate in subj_candidates:
                subj_id = subj_candidate.get("id")
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
                        lit_values.extend(str(val).lower() for val in subj_literals[datatype][predicate])
                        
                lit_string = " ".join(lit_values)
                
                # Calculate token-based similarity
                from alligator.utils import tokenize_text
                lit_tokens = set(tokenize_text(lit_string))
                row_all_tokens = set(tokenize_text(row_text_all))
                row_lit_tokens = set(tokenize_text(row_text_lit))
                
                p_subj_lit_all_datatype = round(len(lit_tokens & row_lit_tokens) / len(lit_tokens | row_lit_tokens) if lit_tokens and row_lit_tokens else 0, 3)
                p_subj_lit_row = round(len(lit_tokens & row_all_tokens) / len(lit_tokens | row_all_tokens) if lit_tokens and row_all_tokens else 0, 3)
                
                subj_candidate["features"]["p_subj_lit_all_datatype"] = p_subj_lit_all_datatype
                subj_candidate["features"]["p_subj_lit_row"] = p_subj_lit_row
                
                # Process each literal column
                for lit_col, lit_type in lit_columns.items():
                    lit_col_int = int(lit_col)
                    if lit_col_int >= len(row):
                        continue
                        
                    lit_value = row[lit_col_int]
                    if not lit_value or pd.isna(lit_value):
                        continue
                        
                    lit_value = str(lit_value).lower()
                    lit_datatype = lit_type.upper()  # Normalize datatype
                    
                    # Find matching datatype in literals
                    normalized_datatype = lit_datatype.lower()
                    if normalized_datatype not in subj_literals:
                        continue
                        
                    # Calculate maximum similarity for this literal column
                    max_score = 0
                    for predicate in subj_literals[normalized_datatype]:
                        for kg_value in subj_literals[normalized_datatype][predicate]:
                            kg_value = str(kg_value).lower()
                            
                            # Calculate similarity based on datatype
                            p_subj_lit = 0
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
                                    p_subj_lit = 0.5 if (lit_value in kg_value or kg_value in lit_value) else 0
                            else:  # STRING
                                # Use token-based similarity
                                lit_tokens = set(tokenize_text(lit_value))
                                kg_tokens = set(tokenize_text(kg_value))
                                if lit_tokens and kg_tokens:
                                    p_subj_lit = len(lit_tokens & kg_tokens) / len(lit_tokens | kg_tokens)
                            
                            p_subj_lit = round(p_subj_lit, 3)
                            
                            if p_subj_lit > 0:
                                # Record match
                                subj_candidate["matches"][lit_col].append({
                                    "p": predicate,
                                    "o": kg_value,
                                    "s": p_subj_lit
                                })
                                
                                # Update maximum score
                                max_score = max(max_score, p_subj_lit)
                                
                                # Update predicates
                                if predicate not in subj_candidate["predicates"][lit_col]:
                                    subj_candidate["predicates"][lit_col][predicate] = 0
                                subj_candidate["predicates"][lit_col][predicate] = max(
                                    subj_candidate["predicates"][lit_col][predicate], p_subj_lit
                                )
                    
                    # Normalize and update feature
                    if max_score > 0:
                        subj_candidate["features"]["p_subj_lit_datatype"] += round(max_score / n_lit_cols, 3)

    def _build_linked_entities_and_training(
        self, ne_columns, row, correct_qids, row_index, candidates_results, bow_data
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
