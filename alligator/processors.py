import hashlib
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Set

import pandas as pd
from pymongo import UpdateOne

from alligator.database import DatabaseAccessMixin
from alligator.feature import Feature
from alligator.fetchers import CandidateFetcher, LiteralFetcher, ObjectFetcher
from alligator.mongo import MongoWrapper
from alligator.typing import Candidate, Entity, RowData
from alligator.utils import ColumnHelper, clean_str


class RowBatchProcessor(DatabaseAccessMixin):
    """
    Extracted logic for process_rows_batch (and associated scoring helpers).
    Takes the Alligator instance so we can reference .mongo_wrapper, .feature, etc.
    """

    def __init__(
        self,
        dataset_name: str,
        table_name: str,
        feature: Feature,
        candidate_fetcher: CandidateFetcher,
        object_fetcher: ObjectFetcher | None = None,
        literal_fetcher: LiteralFetcher | None = None,
        max_candidates_in_result: int = 5,
        fuzzy_retry: bool = False,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.table_name = table_name
        self.feature = feature
        self.candidate_fetcher = candidate_fetcher
        self.object_fetcher = object_fetcher
        self.literal_fetcher = literal_fetcher
        self.max_candidates_in_result = max_candidates_in_result
        self.fuzzy_retry = fuzzy_retry
        self._db_name = kwargs.get("db_name", "alligator_db")
        self._mongo_uri = kwargs.get("mongo_uri", "mongodb://gator-mongodb:27017")
        self.input_collection = kwargs.get("input_collection", "input_data")
        self.candidate_collection = kwargs.get("candidate_collection", "candidates")
        self.mongo_wrapper = MongoWrapper(self._mongo_uri, self._db_name)

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
            candidates = await self._fetch_all_candidates(entities)

            # 3) Process each row and update DB
            await self._process_rows(row_data_list, candidates)

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
            for col_idx, value in enumerate(row):
                row[col_idx] = clean_str(value)
            ne_columns = doc["classified_columns"].get("NE", {})
            lit_columns = doc["classified_columns"].get("LIT", {})
            context_columns = doc.get("context_columns", [])
            correct_qids = doc.get("correct_qids", {})
            row_index = doc.get("row_id")

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
                normalized_col = ColumnHelper.normalize(col_idx)
                if not ColumnHelper.is_valid_index(normalized_col, len(row)):
                    continue

                cell_value = row[ColumnHelper.to_int(normalized_col)]
                # Skip empty or NA values
                if not cell_value or pd.isna(cell_value):
                    continue

                # Normalize value
                qids = correct_qids.get(f"{row_index}-{normalized_col}", [])

                # Create entity
                entity = Entity(
                    value=cell_value,
                    row_index=row_index,
                    col_index=normalized_col,
                    context_text=context_text,
                    correct_qids=qids,
                    fuzzy=False,
                )
                entities.append(entity)

        return entities, row_data_list

    async def _fetch_all_candidates(self, entities: List[Entity]) -> Dict[str, List[Candidate]]:
        """
        Fetch candidates for all entities, with fuzzy retry for poor results.
        """
        # Initial fetch
        initial_results = await self.candidate_fetcher.fetch_candidates_batch(
            entities=[e.value for e in entities],
            fuzzies=[e.fuzzy for e in entities],
            qids=[e.correct_qids if e.correct_qids is not None else [] for e in entities],
        )

        # Find entities needing fuzzy retry
        retry_entities: List[Entity] = []
        for entity in entities:
            candidates = initial_results.get(entity.value, [])
            if self.fuzzy_retry and len(candidates) <= 1:
                # Create a copy of the entity with fuzzy=True
                retry_entity = Entity(
                    value=entity.value,
                    row_index=entity.row_index,
                    col_index=entity.col_index,
                    context_text=entity.context_text,
                    correct_qids=entity.correct_qids,
                    fuzzy=True,
                )
                retry_entities.append(retry_entity)

        # Perform fuzzy retry if needed
        if retry_entities:
            retry_results = await self.candidate_fetcher.fetch_candidates_batch(
                entities=[e.value for e in retry_entities],
                fuzzies=[e.fuzzy for e in retry_entities],
                qids=[
                    e.correct_qids if e.correct_qids is not None else [] for e in retry_entities
                ],
            )

            # Update with retry results
            for entity in retry_entities:
                if entity.value in retry_results:
                    initial_results[entity.value] = retry_results[entity.value]

        candidates: Dict[str, List[Candidate]] = defaultdict(list)
        for mention, candidates_list in initial_results.items():
            for candidate in candidates_list:
                candidate_dict: Dict[str, Any] = {"features": {}}
                for key, value in candidate.items():
                    if key in self.feature.selected_features:
                        candidate_dict["features"][key] = value
                    else:
                        candidate_dict[key] = value
                candidates[mention].append(Candidate.from_dict(candidate_dict))
        return candidates

    async def _process_rows(
        self, row_data_list: List[RowData], candidates: Dict[str, List[Candidate]]
    ):
        bulk_cand = []
        bulk_input = []
        db = self.get_db()

        """Process each row and update database."""
        for row_data in row_data_list:
            entity_ids = set()
            candidates_by_col = {}
            row_value = " ".join(str(v) for v in row_data.row)
            for col_idx, ner_type in row_data.ne_columns.items():
                normalized_col = ColumnHelper.normalize(col_idx)
                if not ColumnHelper.is_valid_index(normalized_col, len(row_data.row)):
                    continue

                cell_value = row_data.row[ColumnHelper.to_int(normalized_col)]
                if not cell_value or pd.isna(cell_value):
                    continue

                entity_value = clean_str(cell_value)
                mention_candidates = candidates.get(entity_value, [])
                if mention_candidates:
                    candidates_by_col[normalized_col] = mention_candidates
                    for cand in mention_candidates:
                        if cand.id:
                            entity_ids.add(cand.id)

                # Process each entity in the row
                self._compute_features(entity_value, row_value, mention_candidates)

            # Enhance with additional features if possible
            if self.object_fetcher and self.literal_fetcher:
                await self._enhance_with_lamapi_features(row_data, entity_ids, candidates_by_col)

            # Update the status in the input collection
            bulk_input.append(
                UpdateOne(
                    {"_id": row_data.doc_id},
                    {
                        "$set": {
                            "status": "DONE",
                            "rank_status": "TODO",
                            "rerank_status": "TODO",
                        }
                    },
                )
            )

            # Store candidates in separate normalized documents
            for col_id, col_candidates in candidates_by_col.items():
                # Create one document per column
                bulk_cand.append(
                    UpdateOne(
                        {
                            "row_id": str(row_data.row_index),
                            "col_id": str(col_id),
                            "owner_id": row_data.doc_id,
                        },
                        {"$set": {"candidates": [c.to_dict() for c in col_candidates]}},
                        upsert=True,
                    )
                )
        bulk_batch_size = 8192
        if bulk_cand:
            for i in range(0, len(bulk_cand), bulk_batch_size):
                db[self.candidate_collection].bulk_write(
                    bulk_cand[i : i + bulk_batch_size], ordered=False
                )

        if bulk_input:
            for i in range(0, len(bulk_input), bulk_batch_size):
                db[self.input_collection].bulk_write(
                    bulk_input[i : i + bulk_batch_size], ordered=False
                )

    def _compute_features(self, entity_value: str, row_value: str, candidates: List[Candidate]):
        """Process entities by computing features. Feature computation
        is done in-place over the candidates."""

        self.feature.process_candidates(candidates, entity_value, row_value)

    async def _enhance_with_lamapi_features(
        self,
        row_data: RowData,
        entity_ids: Set[str],
        candidates_by_col: Dict[str, List[Candidate]],
    ):
        """Enhance candidates with LAMAPI features."""

        # Fetch external data
        objects_data = None
        if self.object_fetcher:
            objects_data = await self.object_fetcher.fetch_objects(list(entity_ids))

        literals_data = None
        if self.literal_fetcher:
            literals_data = await self.literal_fetcher.fetch_literals(list(entity_ids))

        if objects_data is not None:
            self.feature.compute_entity_entity_relationships(candidates_by_col, objects_data)

        if literals_data is not None:
            self.feature.compute_entity_literal_relationships(
                candidates_by_col, row_data.lit_columns, row_data.row, literals_data
            )
