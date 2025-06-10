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
            cleaned_row = [clean_str(str(cell)) for cell in row]
            ne_columns = doc["classified_columns"].get("NE", {})
            lit_columns = doc["classified_columns"].get("LIT", {})
            context_columns = doc.get("context_columns", [])
            correct_qids = doc.get("correct_qids", {})
            row_index = doc.get("row_id")

            # Create row data
            row_data = RowData(
                doc_id=doc["_id"],
                row=cleaned_row,
                ne_columns=ne_columns,
                lit_columns=lit_columns,
                context_columns=context_columns,
                correct_qids=correct_qids,
                row_index=row_index,
            )
            row_data_list.append(row_data)

            # Extract entities from this row
            for col_idx, ner_type in ne_columns.items():
                normalized_col = ColumnHelper.normalize(col_idx)
                if not ColumnHelper.is_valid_index(normalized_col, len(row)):
                    continue

                cell_value = cleaned_row[ColumnHelper.to_int(normalized_col)]
                if not cell_value or pd.isna(cell_value):
                    continue

                qids = correct_qids.get(f"{row_index}-{normalized_col}", [])
                entity = Entity(
                    value=cell_value,
                    row_index=row_index,
                    col_index=normalized_col,
                    correct_qids=qids,
                    fuzzy=False,
                )
                entities.append(entity)

        return entities, row_data_list

    async def _fetch_all_candidates(self, entities: List[Entity]) -> Dict[str, List[Candidate]]:
        """
        Fetch candidates for all entities, with fuzzy retry for poor results.
        Now optimized to fetch only distinct mentions per batch.
        """
        fetch_groups = set()
        for entity in entities:
            qids_key = tuple(sorted(entity.correct_qids)) if entity.correct_qids else ()
            fetch_key = (entity.value, entity.fuzzy, qids_key)
            fetch_groups.add(fetch_key)

        unique_entities = []
        unique_fuzzies = []
        unique_qids = []
        for value, fuzzy, qids_tuple in fetch_groups:
            unique_entities.append(value)
            unique_fuzzies.append(fuzzy)
            unique_qids.append(list(qids_tuple))

        print(
            f"Fetching candidates for {len(unique_entities)} distinct mentions "
            f"(from {len(entities)} total entities)"
        )

        initial_results = await self.candidate_fetcher.fetch_candidates_batch(
            entities=unique_entities,
            fuzzies=unique_fuzzies,
            qids=unique_qids,
        )

        retry_fetch_groups = set()
        for value, fuzzy, qids_tuple in fetch_groups:
            retrieved_candidates = initial_results.get(value, [])
            if self.fuzzy_retry and not fuzzy and len(retrieved_candidates) < 1:
                retry_key = (value, True, qids_tuple)  # fuzzy=True for retry
                retry_fetch_groups.add(retry_key)

        if retry_fetch_groups:
            retry_entities = []
            retry_fuzzies = []
            retry_qids = []
            for value, fuzzy, qids_tuple in retry_fetch_groups:
                retry_entities.append(value)
                retry_fuzzies.append(fuzzy)
                retry_qids.append(list(qids_tuple))

            print(f"Performing fuzzy retry for {len(retry_entities)} distinct mentions")

            retry_results = await self.candidate_fetcher.fetch_candidates_batch(
                entities=retry_entities,
                fuzzies=retry_fuzzies,
                qids=retry_qids,
            )

            for value in retry_entities:
                if value in retry_results:
                    initial_results[value] = retry_results[value]

        # Convert to Candidate objects (reuse results for all entities with same mention)
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

                mention_candidates = candidates.get(cell_value, [])
                if mention_candidates:
                    candidates_by_col[normalized_col] = mention_candidates
                    for cand in mention_candidates:
                        if cand.id:
                            entity_ids.add(cand.id)

                # Process each entity in the row
                self._compute_features(row_value, mention_candidates)

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

    def _compute_features(self, row_value: str, candidates: List[Candidate]):
        """Process entities by computing features. Feature computation
        is done in-place over the candidates."""

        self.feature.process_candidates(candidates, row_value)

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
