import hashlib
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from alligator.feature import Feature
from alligator.fetchers import CandidateFetcher, LiteralFetcher, ObjectFetcher
from alligator.mongo import MongoWrapper
from alligator.utils import tokenize_text


@dataclass
class Entity:
    """Represents a named entity from a table cell."""

    value: str
    row_index: Optional[int]
    col_index: str  # Stored as string for consistency
    context_text: str
    correct_qid: Optional[str] = None
    fuzzy: bool = False


@dataclass
class RowData:
    """Represents a row's data with all necessary context."""

    doc_id: str
    row: List[Any]
    ne_columns: Dict[str, str]
    lit_columns: Dict[str, str]
    context_columns: List[str]
    correct_qids: Dict[str, str]
    row_index: Optional[int]
    context_text: str
    row_hash: str


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
          1) Extract entities and row data
          2) Fetch and enhance candidates
          3) Process and save results
        """
        try:
            # 1) Extract entities and row data
            entities, row_data_list = self._extract_entities(docs)

            # 2) Fetch initial candidates in one batch
            candidates_results = await self._fetch_all_candidates(entities)

            # 3) Process each row and update DB
            await self._process_rows(row_data_list, candidates_results)

        except Exception:
            self.mongo_wrapper.log_to_db(
                "ERROR", "Error processing batch of rows", traceback.format_exc()
            )

    def _extract_entities(self, docs) -> tuple[List[Entity], List[RowData]]:
        """
        Extract entities and row data from documents.
        """
        entities = []
        row_data_list = []

        for doc in docs:
            row = doc["data"]
            ne_columns = doc["classified_columns"].get("NE", {})
            context_columns = doc.get("context_columns", [])
            correct_qids = doc.get("correct_qids", {})
            row_index = doc.get("row_id")
            lit_columns = doc["classified_columns"].get("LIT", {})

            # Build context text
            context_text = " ".join(str(row[int(c)]) for c in sorted(context_columns, key=int))
            normalized_text = " ".join(context_text.lower().split())
            row_hash = hashlib.sha256(normalized_text.encode()).hexdigest()

            # Create row data
            row_data = RowData(
                doc_id=doc["_id"],
                row=row,
                ne_columns=ne_columns,
                lit_columns=lit_columns,
                context_columns=context_columns,
                correct_qids=correct_qids,
                row_index=row_index,
                context_text=context_text,
                row_hash=row_hash,
            )
            row_data_list.append(row_data)

            # Extract entities from this row
            for col_idx, ner_type in ne_columns.items():
                col_idx_str = str(col_idx)
                col_idx_int = int(col_idx_str)

                # Skip if column index is out of bounds
                if col_idx_int >= len(row):
                    continue

                cell_value = row[col_idx_int]
                # Skip empty or NA values
                if not cell_value or pd.isna(cell_value):
                    continue

                # Normalize value
                normalized_value = str(cell_value).strip().replace("_", " ").lower()
                correct_qid = correct_qids.get(f"{row_index}-{col_idx_str}")

                # Create entity
                entity = Entity(
                    value=normalized_value,
                    row_index=row_index,
                    col_index=col_idx_str,
                    context_text=context_text,
                    correct_qid=correct_qid,
                    fuzzy=False,
                )
                entities.append(entity)

        return entities, row_data_list

    async def _fetch_all_candidates(self, entities: List[Entity]) -> Dict[str, List[dict]]:
        """
        Fetch candidates for all entities, with fuzzy retry for poor results.
        """
        # Initial fetch
        initial_results = await self.candidate_fetcher.fetch_candidates_batch(
            entities=[e.value for e in entities],
            row_texts=[e.context_text for e in entities],
            fuzzies=[e.fuzzy for e in entities],
            qids=[e.correct_qid for e in entities],
        )

        # Find entities needing fuzzy retry
        retry_entities = []
        for entity in entities:
            candidates = initial_results.get(entity.value, [])
            if len(candidates) <= 1:
                # Create a copy of the entity with fuzzy=True
                retry_entity = Entity(
                    value=entity.value,
                    row_index=entity.row_index,
                    col_index=entity.col_index,
                    context_text=entity.context_text,
                    correct_qid=entity.correct_qid,
                    fuzzy=True,
                )
                retry_entities.append(retry_entity)

        # Perform fuzzy retry if needed
        if retry_entities:
            retry_results = await self.candidate_fetcher.fetch_candidates_batch(
                entities=[e.value for e in retry_entities],
                row_texts=[e.context_text for e in retry_entities],
                fuzzies=[e.fuzzy for e in retry_entities],
                qids=[e.correct_qid for e in retry_entities],
            )

            # Update with retry results
            for entity in retry_entities:
                if entity.value in retry_results:
                    initial_results[entity.value] = retry_results[entity.value]

        return initial_results

    async def _process_rows(
        self, row_data_list: List[RowData], candidates_results: Dict[str, List[dict]]
    ):
        db = self.get_db()

        """Process each row and update database."""
        for row_data in row_data_list:
            # Collect QIDs from this row (needed for relationship features)
            self._collect_row_qids(row_data, candidates_results)

            # Process each entity in the row
            self._process_row_entities(row_data, candidates_results)

            # Enhance with additional features if possible
            if self.object_fetcher and self.literal_fetcher:
                await self._enhance_with_lamapi_features(row_data, candidates_results)

            # Build final results
            training_candidates = self._build_linked_entities_and_training(
                row_data, candidates_results
            )

            # Update database
            db[self.input_collection].update_one(
                {"_id": row_data.doc_id},
                {
                    "$set": {
                        "candidates": training_candidates,
                        "status": "DONE",
                        "rank_status": "TODO",
                        "rerank_status": "TODO",
                    }
                },
            )

    def _process_row_entities(self, row_data: RowData, candidates_results: Dict[str, List[dict]]):
        """Process entities in a row by computing features."""
        for col_idx, ner_type in row_data.ne_columns.items():
            col_idx_int = int(col_idx)

            if col_idx_int >= len(row_data.row):
                continue

            cell_value = row_data.row[col_idx_int]
            if not cell_value or pd.isna(cell_value):
                continue

            # Normalize value
            entity_value = str(cell_value).strip().replace("_", " ").lower()
            candidates = candidates_results.get(entity_value, [])

            # Process candidates with features
            row_tokens = set(tokenize_text(entity_value))
            candidates_results[entity_value] = self.feature.process_candidates(
                candidates, entity_value, row_tokens
            )

    def _collect_row_qids(
        self, row_data: RowData, candidates_results: Dict[str, List[dict]]
    ) -> List[str]:
        """Collect all QIDs from a row's candidates."""
        row_qids = set()

        for col_idx, ner_type in row_data.ne_columns.items():
            col_idx_int = int(col_idx)

            if col_idx_int >= len(row_data.row):
                continue

            cell_value = row_data.row[col_idx_int]
            if not cell_value or pd.isna(cell_value):
                continue

            # Normalize value
            entity_value = str(cell_value).strip().replace("_", " ").lower()
            candidates = candidates_results.get(entity_value, [])

            # Collect QIDs
            for cand in candidates:
                if cand.get("id"):
                    row_qids.add(cand["id"])

        return list(row_qids)

    async def _enhance_with_lamapi_features(
        self, row_data: RowData, candidates_results: Dict[str, List[dict]]
    ):
        """Enhance candidates with LAMAPI features."""
        # Collect all candidates by column
        all_candidates_by_col = {}
        all_entity_ids = set()

        for col_idx, ner_type in row_data.ne_columns.items():
            col_idx_int = int(col_idx)

            if col_idx_int >= len(row_data.row):
                continue

            cell_value = row_data.row[col_idx_int]
            if not cell_value or pd.isna(cell_value):
                continue

            # Normalize value
            entity_value = str(cell_value).strip().replace("_", " ").lower()
            candidates = candidates_results.get(entity_value, [])

            if candidates:
                all_candidates_by_col[col_idx] = candidates
                for cand in candidates:
                    if cand.get("id"):
                        all_entity_ids.add(cand["id"])
                        # Initialize tracking structures
                        cand.setdefault("matches", defaultdict(list))
                        cand.setdefault("predicates", defaultdict(dict))

        if not all_entity_ids:
            return

        # Fetch external data
        objects_data = await self.object_fetcher.fetch_objects(list(all_entity_ids))
        literals_data = await self.literal_fetcher.fetch_literals(list(all_entity_ids))

        if not objects_data and not literals_data:
            return

        # Process entity-entity relationships
        self.feature.compute_entity_entity_relationships(all_candidates_by_col, objects_data)

        # Process entity-literal relationships
        self.feature.compute_entity_literal_relationships(
            all_candidates_by_col, row_data.lit_columns, row_data.row, literals_data
        )

    def _build_linked_entities_and_training(
        self, row_data: RowData, candidates_results: Dict[str, List[dict]]
    ) -> Dict[str, List[dict]]:
        """Build final ranked candidate lists by column."""
        training_candidates_by_col = {}

        for col_idx, ner_type in row_data.ne_columns.items():
            col_idx_int = int(col_idx)

            if col_idx_int >= len(row_data.row):
                continue

            cell_value = row_data.row[col_idx_int]
            if not cell_value or pd.isna(cell_value):
                continue

            # Normalize value
            entity_value = str(cell_value).strip().replace("_", " ").lower()
            candidates = candidates_results.get(entity_value, [])

            # Rank candidates
            max_training_candidates = len(candidates)
            ranked_candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)

            # Handle correct QID if provided
            correct_qid = row_data.correct_qids.get(f"{row_data.row_index}-{col_idx}")
            if correct_qid:
                # Check if correct QID is in top results
                top_ids = [
                    c["id"] for c in ranked_candidates[:max_training_candidates] if c.get("id")
                ]
                if correct_qid not in top_ids:
                    # Try to find correct candidate
                    correct_candidate = next(
                        (c for c in ranked_candidates if c.get("id") == correct_qid), None
                    )
                    if correct_candidate:
                        # Replace last candidate with correct one
                        top_candidates = ranked_candidates[: max_training_candidates - 1]
                        top_candidates.append(correct_candidate)
                        ranked_candidates = sorted(
                            top_candidates, key=lambda x: x.get("score", 0.0), reverse=True
                        )

            # Store final results
            training_candidates = ranked_candidates[:max_training_candidates]
            if training_candidates:
                training_candidates_by_col[col_idx] = training_candidates

        return training_candidates_by_col
