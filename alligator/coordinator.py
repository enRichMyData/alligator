"""
Coordinator and manager classes for the Alligator entity linking system.

This module implements the coordinator pattern to orchestrate the entity linking
pipeline through specialized managers, replacing the monolithic Alligator class.
"""

import asyncio
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import aiohttp
import pandas as pd
from column_classifier import ColumnClassifier

from alligator import TIMEOUT
from alligator.config import AlligatorConfig
from alligator.database import DatabaseAccessMixin
from alligator.feature import Feature
from alligator.fetchers import CandidateFetcher, LiteralFetcher, ObjectFetcher
from alligator.ml import MLWorker
from alligator.mongo import MongoWrapper
from alligator.processors import RowBatchProcessor


class DataManager(DatabaseAccessMixin):
    """Manages data onboarding, validation, and classification."""

    def __init__(self, config: AlligatorConfig):
        self.config = config
        self._mongo_uri = config.database.mongo_uri or "mongodb://gator-mongodb:27017/"
        self._db_name = config.database.db_name or "alligator_db"
        self.mongo_wrapper = MongoWrapper(
            self._mongo_uri,
            self._db_name,
            config.database.input_collection,
            config.database.error_log_collection,
        )
        self.mongo_wrapper.create_indexes()

    def onboard_data(self) -> None:
        """Efficiently load data into MongoDB using batched inserts."""
        start_time = time.perf_counter()

        # Get database connection
        db = self.get_db()
        input_collection = db[self.config.database.input_collection]

        # Ensure we have valid dataset and table names
        dataset_name = self.config.data.dataset_name or "default_dataset"
        table_name = self.config.data.table_name or "default_table"

        # Step 1: Determine data source and extract sample for classification
        if isinstance(self.config.data.input_csv, pd.DataFrame):
            df = self.config.data.input_csv
            sample = df
            total_rows = len(df)
            is_csv_path = False
        else:
            if self.config.data.input_csv is None:
                raise ValueError("Input CSV path cannot be None")
            sample = pd.read_csv(self.config.data.input_csv, nrows=32)
            total_rows = -1
            is_csv_path = True

        print(f"Onboarding {total_rows} rows for dataset '{dataset_name}', table '{table_name}'")

        # Step 2: Perform column classification
        classified_columns = self._classify_columns(sample)
        ne_cols, lit_cols, ignored_cols, context_cols = self._process_column_types(
            sample, classified_columns
        )

        # Step 3: Process all chunks using the generator
        processed_rows = self._process_data_chunks(
            input_collection,
            ne_cols,
            lit_cols,
            ignored_cols,
            context_cols,
            is_csv_path,
            total_rows,
            start_time,
            dataset_name,
            table_name,
        )

        total_time = time.perf_counter() - start_time
        print(
            f"Data onboarding complete for dataset '{dataset_name}' "
            f"and table '{table_name}' - {processed_rows} rows in {total_time:.1f}s"
        )

    def _classify_columns(self, sample: pd.DataFrame) -> Dict[str, str]:
        """Classify columns using the column classifier."""
        classifier = ColumnClassifier(model_type="fast")
        classification_results = classifier.classify_multiple_tables([sample])
        table_classification = classification_results[0].get("table_1", {})

        classified_columns = {}
        for idx, col in enumerate(sample.columns):
            col_result = table_classification.get(col, {})
            classification = col_result.get("classification", "UNKNOWN")
            classified_columns[str(idx)] = classification

        return classified_columns

    def _process_column_types(
        self, sample: pd.DataFrame, classified_columns: Dict[str, str]
    ) -> Tuple[Dict[str, str], Dict[str, str], List[str], List[str]]:
        """Process column classifications into NE, LIT, and ignored columns."""
        ne_cols: Dict[str, str] = {}
        lit_cols: Dict[str, str] = {}
        ignored_cols: List[str] = []

        ne_types = {"PERSON", "OTHER", "ORGANIZATION", "LOCATION"}
        lit_types = {"NUMBER", "DATE", "STRING"}

        for idx, col in enumerate(sample.columns):
            classification = classified_columns.get(str(idx), "UNKNOWN")
            if classification in ne_types:
                ne_cols[str(idx)] = classification
            elif classification in lit_types:
                if classification == "DATE":
                    classification = "DATETIME"
                lit_cols[str(idx)] = classification

        # Override with target columns if provided
        if self.config.data.target_columns is not None:
            target_ne = self.config.data.target_columns.get("NE", {})
            if target_ne:
                ne_cols = cast(Dict[str, str], target_ne)
                for col in ne_cols:
                    if col not in classified_columns:
                        ne_cols[col] = classified_columns.get(col, "UNKNOWN")

            target_lit = self.config.data.target_columns.get("LIT", {})
            if target_lit:
                lit_cols = cast(Dict[str, str], target_lit)
                for col in lit_cols:
                    if not lit_cols[col]:
                        lit_cols[col] = classified_columns.get(col, "UNKNOWN")

            target_ignored = self.config.data.target_columns.get("IGNORED", [])
            if target_ignored:
                ignored_cols = target_ignored

        # Calculate context columns
        all_recognized_cols = set(ne_cols.keys()) | set(lit_cols.keys())
        all_cols = set([str(i) for i in range(len(sample.columns))])
        if len(all_recognized_cols) != len(all_cols):
            ignored_cols.extend(list(all_cols - all_recognized_cols))
        ignored_cols = list(set(ignored_cols))
        context_cols = list(set([str(i) for i in range(len(sample.columns))]) - set(ignored_cols))
        context_cols = sorted(context_cols, key=lambda x: int(x))

        return ne_cols, lit_cols, ignored_cols, context_cols

    def _get_data_chunks(self, is_csv_path: bool, total_rows: int):
        """Generator that yields chunks of rows, handling both DF and CSV."""
        if self.config.data.dry_run:
            if is_csv_path:
                if self.config.data.input_csv is not None:
                    yield pd.read_csv(self.config.data.input_csv, nrows=1), 0
            else:
                if isinstance(self.config.data.input_csv, pd.DataFrame):
                    yield self.config.data.input_csv.iloc[:1], 0
        else:
            if is_csv_path:
                chunk_size = 2048
                row_count = 0
                if self.config.data.input_csv is not None:
                    for chunk in pd.read_csv(self.config.data.input_csv, chunksize=chunk_size):
                        yield chunk, row_count
                        row_count += len(chunk)
            else:
                if isinstance(self.config.data.input_csv, pd.DataFrame):
                    chunk_size = (
                        1024 if total_rows > 100000 else 2048 if total_rows > 10000 else 4096
                    )
                    total_chunks = (total_rows + chunk_size - 1) // chunk_size
                    for chunk_idx in range(total_chunks):
                        chunk_start = chunk_idx * chunk_size
                        chunk_end = min(chunk_start + chunk_size, total_rows)
                        yield self.config.data.input_csv.iloc[chunk_start:chunk_end], chunk_start

    def _process_data_chunks(
        self,
        input_collection,
        ne_cols: Dict[str, str],
        lit_cols: Dict[str, str],
        ignored_cols: List[str],
        context_cols: List[str],
        is_csv_path: bool,
        total_rows: int,
        start_time: float,
        dataset_name: str,
        table_name: str,
    ) -> int:
        """Process data chunks and insert into MongoDB."""
        processed_rows = 0
        chunk_idx = 0

        for chunk, start_idx in self._get_data_chunks(is_csv_path, total_rows):
            chunk_idx += 1
            documents = []

            for i, (_, row) in enumerate(chunk.iterrows()):
                row_id = start_idx + i
                if (
                    str(row_id) not in self.config.data.target_rows
                    and self.config.data.target_rows
                ):
                    continue

                document = {
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                    "row_id": str(row_id),
                    "data": row.tolist(),
                    "classified_columns": {
                        "NE": ne_cols,
                        "LIT": lit_cols,
                        "IGNORED": ignored_cols,
                    },
                    "context_columns": context_cols,
                    "status": "TODO",
                }

                # Add correct QIDs if available
                correct_qids = {}
                for col_id, _ in ne_cols.items():
                    key = f"{row_id}-{col_id}"
                    if key in self.config.data.correct_qids:
                        correct_qids[key] = self.config.data.correct_qids[key]
                        if isinstance(correct_qids[key], str):
                            correct_qids[key] = [correct_qids[key]]
                        else:
                            correct_qids[key] = list(set(correct_qids[key]))
                document["correct_qids"] = correct_qids
                documents.append(document)

            if documents:
                try:
                    input_collection.insert_many(documents, ordered=False)
                    chunk_size = len(documents)
                    processed_rows += chunk_size
                    elapsed = time.perf_counter() - start_time
                    rows_per_second = processed_rows / elapsed if elapsed > 0 else 0

                    if is_csv_path:
                        print(
                            f"Chunk {chunk_idx}: Processed {chunk_size} rows "
                            f"(total: {processed_rows}) ({rows_per_second:.1f} rows/sec)"
                        )
                    else:
                        chunk_start = start_idx + 1
                        chunk_end = start_idx + chunk_size
                        total_chunks = (total_rows + chunk_size - 1) // chunk_size
                        print(
                            f"Chunk {chunk_idx}/{total_chunks}: "
                            f"Onboarded rows {chunk_start}-{chunk_end} "
                            f"({rows_per_second:.1f} rows/sec)"
                        )
                except Exception as e:
                    print(f"Error inserting batch {chunk_idx}: {str(e)}")
                    if "duplicate key" not in str(e).lower():
                        raise

        return processed_rows


class WorkerManager(DatabaseAccessMixin):
    """Manages worker processes for batch processing."""

    def __init__(self, config: AlligatorConfig):
        self.config = config
        self._mongo_uri = config.database.mongo_uri or "mongodb://gator-mongodb:27017/"
        self._db_name = config.database.db_name or "alligator_db"

    async def initialize_async_components(self) -> aiohttp.ClientSession:
        """Initialize components that require an active event loop."""
        connector = aiohttp.TCPConnector(
            limit=self.config.retrieval.http_session_limit,
            ssl=self.config.retrieval.http_session_ssl_verify,
        )
        session = aiohttp.ClientSession(connector=connector, timeout=TIMEOUT)
        print(
            f"Created HTTP session with limit {self.config.retrieval.http_session_limit} "
            f"and SSL verify {self.config.retrieval.http_session_ssl_verify}."
        )
        return session

    def create_fetchers_and_processor(self, session: aiohttp.ClientSession, feature: Feature):
        """Create fetcher and processor instances."""
        # Ensure we have required endpoints
        entity_endpoint = self.config.retrieval.entity_retrieval_endpoint or ""
        entity_token = self.config.retrieval.entity_retrieval_token or ""
        db_name = self.config.database.db_name or "alligator_db"
        mongo_uri = self.config.database.mongo_uri or "mongodb://gator-mongodb:27017/"
        dataset_name = self.config.data.dataset_name or "default_dataset"
        table_name = self.config.data.table_name or "default_table"

        candidate_fetcher = CandidateFetcher(
            entity_endpoint,
            entity_token,
            self.config.retrieval.candidate_retrieval_limit,
            feature,
            session=session,
            db_name=db_name,
            mongo_uri=mongo_uri,
            input_collection=self.config.database.input_collection,
            cache_collection=self.config.database.cache_collection,
        )

        object_fetcher = None
        if self.config.retrieval.object_retrieval_endpoint:
            object_fetcher = ObjectFetcher(
                self.config.retrieval.object_retrieval_endpoint,
                entity_token,
                session=session,
                db_name=db_name,
                mongo_uri=mongo_uri,
                cache_collection=self.config.database.object_cache_collection,
            )

        literal_fetcher = None
        if self.config.retrieval.literal_retrieval_endpoint:
            literal_fetcher = LiteralFetcher(
                self.config.retrieval.literal_retrieval_endpoint,
                entity_token,
                session=session,
                db_name=db_name,
                mongo_uri=mongo_uri,
                cache_collection=self.config.database.literal_cache_collection,
            )

        row_processor = RowBatchProcessor(
            dataset_name,
            table_name,
            feature,
            candidate_fetcher,
            object_fetcher,
            literal_fetcher,
            self.config.retrieval.max_candidates_in_result,
            db_name=db_name,
            mongo_uri=mongo_uri,
            input_collection=self.config.database.input_collection,
        )

        return candidate_fetcher, object_fetcher, literal_fetcher, row_processor

    def run_workers(self, feature: Feature) -> None:
        """Run worker processes for batch processing."""
        db = self.get_db()
        input_collection = db[self.config.database.input_collection]

        mongo_wrapper = MongoWrapper(
            self._mongo_uri,
            self._db_name,
            self.config.database.input_collection,
            self.config.database.error_log_collection,
        )

        total_rows = mongo_wrapper.count_documents(input_collection, {"status": "TODO"})
        print(f"Found {total_rows} tasks to process.")

        num_workers = self.config.worker.num_workers or 1
        processes = []
        for rank in range(num_workers):
            p = mp.Process(target=self._worker, args=(rank, feature))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            p.close()

    def _worker(self, rank: int, feature: Feature):
        """Worker process entry point."""
        asyncio.run(self._worker_async(rank, feature))

    async def _worker_async(self, rank: int, feature: Feature):
        """Async worker process implementation."""
        print(f"Worker {rank} started.")
        db = self.get_db()
        input_collection = db[self.config.database.input_collection]

        session = await self.initialize_async_components()
        (
            candidate_fetcher,
            object_fetcher,
            literal_fetcher,
            row_processor,
        ) = self.create_fetchers_and_processor(session, feature)

        total_docs = input_collection.count_documents(
            {
                "dataset_name": self.config.data.dataset_name,
                "table_name": self.config.data.table_name,
            }
        )
        docs_per_worker = total_docs // self.config.worker.num_workers
        remainder = total_docs % self.config.worker.num_workers

        # Adjust batch size for this specific worker
        # Give extra documents to earlier workers if not evenly divisible
        if rank < remainder:
            batch_size = docs_per_worker + 1
            skip = rank * (docs_per_worker + 1)
        else:
            batch_size = docs_per_worker
            skip = (remainder * (docs_per_worker + 1)) + ((rank - remainder) * docs_per_worker)
        if batch_size <= 0:
            print(f"Worker {rank} has no documents to process (batch size is {batch_size}).")
            await session.close()
            return
        print(f"Worker {rank} started with batch size {batch_size}, skipping {skip}.")

        # Find documents to process for this worker using skip/limit
        cursor = (
            input_collection.find(
                {
                    "dataset_name": self.config.data.dataset_name,
                    "table_name": self.config.data.table_name,
                }
            )
            .skip(skip)
            .limit(batch_size)
        )

        todo_docs = []
        for doc in cursor:
            input_collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "DOING"}})
            if len(todo_docs) < self.config.worker.worker_batch_size:
                todo_docs.append(doc)
            else:
                print(f"Worker {rank} processing batch of {len(todo_docs)} documents.")
                await self._process_batch(todo_docs, row_processor)
                print(f"Worker {rank} processed {len(todo_docs)} documents.")
                todo_docs = [doc]
        if todo_docs:
            print(f"Worker {rank} processing final batch of {len(todo_docs)} documents.")
            await self._process_batch(todo_docs, row_processor)
            print(f"Worker {rank} finished processing documents.")

        await session.close()

    async def _process_batch(self, docs: List[Dict], row_processor: RowBatchProcessor):
        """Process a batch of documents."""
        if row_processor:
            await row_processor.process_rows_batch(docs)


class MLManager:
    """Manages machine learning pipeline for ranking and reranking."""

    def __init__(self, config: AlligatorConfig):
        self.config = config

    def run_ml_pipeline(self, feature: Feature) -> None:
        """Run the complete ML pipeline with ranking and reranking stages."""
        num_workers = self.config.worker.num_workers or 1
        pool = mp.Pool(processes=num_workers)

        try:
            # First ML ranking stage
            rank_stage_partial_func = partial(
                self._ml_worker,
                stage="rank",
                global_frequencies=(None, None, None),
            )
            pool.map(rank_stage_partial_func, range(self.config.ml.num_ml_workers))
            print("ML Rank stage complete.")

            # Compute global frequencies (this happens in the main process)
            print("Computing global frequencies...")
            (
                type_frequencies,
                predicate_frequencies,
                predicate_pair_frequencies,
            ) = feature.compute_global_frequencies(
                docs_to_process=self.config.feature.doc_percentage_type_features,
                random_sample=False,
            )
            print("Global frequencies computed.")

            # Second ML ranking stage with global frequencies
            rerank_stage_partial_func = partial(
                self._ml_worker,
                stage="rerank",
                global_frequencies=(
                    type_frequencies,
                    predicate_frequencies,
                    predicate_pair_frequencies,
                ),
            )
            pool.map(rerank_stage_partial_func, range(self.config.ml.num_ml_workers))
            print("ML Rerank stage complete.")

        finally:
            pool.close()
            pool.join()

    def _ml_worker(self, rank: int, stage: str, global_frequencies: Tuple):
        """ML worker process implementation."""
        dataset_name = self.config.data.dataset_name or "default_dataset"
        table_name = self.config.data.table_name or "default_table"
        mongo_uri = self.config.database.mongo_uri or "mongodb://gator-mongodb:27017/"
        db_name = self.config.database.db_name or "alligator_db"

        # Determine model path based on stage
        model_path = (
            self.config.ml.ranker_model_path
            if stage == "rank"
            else self.config.ml.reranker_model_path
        )
        max_candidates_for_stage = (
            -1 if stage == "rank" else self.config.retrieval.max_candidates_in_result
        )

        ml_worker = MLWorker(
            rank,
            table_name=table_name,
            dataset_name=dataset_name,
            stage=stage,
            model_path=model_path,
            batch_size=self.config.ml.ml_worker_batch_size,
            max_candidates_in_result=max_candidates_for_stage,
            top_n_cta_cpa_freq=self.config.feature.top_n_cta_cpa_freq,
            features=self.config.ml.selected_features,
            mongo_uri=mongo_uri,
            db_name=db_name,
            input_collection=self.config.database.input_collection,
        )

        # Run the ML worker with global frequencies
        return ml_worker.run(global_frequencies=global_frequencies)


class OutputManager(DatabaseAccessMixin):
    """Manages output generation and saving."""

    def __init__(self, config: AlligatorConfig):
        self.config = config
        self._mongo_uri = config.database.mongo_uri or "mongodb://gator-mongodb:27017/"
        self._db_name = config.database.db_name or "alligator_db"

    def save_output(self) -> List[Dict[str, Any]]:
        """Save output to CSV and return extracted rows."""
        if not self.config.data.save_output:
            return [{}]

        print("Saving output...")
        db = self.get_db()
        input_collection = db[self.config.database.input_collection]

        dataset_name = self.config.data.dataset_name or "default_dataset"
        table_name = self.config.data.table_name or "default_table"

        header = None
        if isinstance(self.config.data.input_csv, pd.DataFrame):
            header = self.config.data.input_csv.columns.tolist()
        elif isinstance(self.config.data.input_csv, (str, Path)):
            header = pd.read_csv(self.config.data.input_csv, nrows=0).columns.tolist()

        # Get first document to determine column count if header is still None
        sample_doc = input_collection.find_one(
            {"dataset_name": dataset_name, "table_name": table_name}
        )
        if not sample_doc:
            print("No documents found for the specified dataset and table.")
            return []

        if header is None:
            print("Could not extract header from input table, using generic column names.")
            header = [f"col_{i}" for i in range(len(sample_doc["data"]))]

        # Write directly to CSV without storing in memory
        if self.config.data.save_output_to_csv and isinstance(
            self.config.data.output_csv, (str, Path)
        ):
            first_row = True
            with open(self.config.data.output_csv, "w", newline="", encoding="utf-8") as csvfile:
                writer = None
                for row_data in self.document_generator(header):
                    if first_row:
                        import csv

                        writer = csv.DictWriter(csvfile, fieldnames=row_data.keys())
                        writer.writeheader()
                        first_row = False

                    writer.writerow(row_data)
            return [{}]
        else:
            return list(self.document_generator(header))

    def document_generator(self, header: List[str]):
        dg = self.get_db()
        input_collection = dg[self.config.database.input_collection]
        cursor = input_collection.find(
            {
                "dataset_name": self.config.data.dataset_name,
                "table_name": self.config.data.table_name,
            },
            projection={"data": 1, "cea": 1, "classified_columns.NE": 1},
        ).batch_size(512)
        for doc in cursor:
            yield self._extract_row_data(doc, header)

    def _extract_row_data(self, doc, header):
        """Extract row data from a MongoDB document.

        Encapsulates the common logic for formatting a row from a document.
        """
        # Create base row data with original values
        row_data = dict(zip(header, doc["data"]))
        el_results = doc.get("cea", {})

        # Add entity linking results
        for col_idx, col_type in doc["classified_columns"].get("NE", {}).items():
            col_index = int(col_idx)
            col_header = header[col_index]

            id_field = f"{col_header}_id"
            name_field = f"{col_header}_name"
            desc_field = f"{col_header}_desc"
            score_field = f"{col_header}_score"

            # Get first candidate or empty placeholder
            candidate = el_results.get(col_idx, [{}])[0]

            row_data[id_field] = candidate.get("id", "")
            row_data[name_field] = candidate.get("name", "")
            row_data[desc_field] = candidate.get("description", "")
            row_data[score_field] = candidate.get("score", 0)

        return row_data


class AlligatorCoordinator:
    """
    Main coordinator that orchestrates the entity linking pipeline.

    This class coordinates the different managers to execute the complete
    entity linking workflow while maintaining clean separation of concerns.
    """

    def __init__(self, config: AlligatorConfig):
        self.config = config

        # Initialize managers
        self.data_manager = DataManager(config)
        self.worker_manager = WorkerManager(config)
        self.ml_manager = MLManager(config)
        self.output_manager = OutputManager(config)

        # Initialize feature computation
        dataset_name = config.data.dataset_name or "default_dataset"
        table_name = config.data.table_name or "default_table"
        db_name = config.database.db_name or "alligator_db"
        mongo_uri = config.database.mongo_uri or "mongodb://gator-mongodb:27017/"

        self.feature = Feature(
            dataset_name,
            table_name,
            top_n_cta_cpa_freq=config.feature.top_n_cta_cpa_freq,
            features=config.ml.selected_features,
            db_name=db_name,
            mongo_uri=mongo_uri,
            input_collection=config.database.input_collection,
        )

    def run(self) -> List[Dict[str, Any]]:
        """Execute the complete entity linking pipeline."""
        print("Starting Alligator entity linking pipeline...")

        # Step 1: Data onboarding
        print("Step 1: Data onboarding...")
        self.data_manager.onboard_data()

        # Step 2: Worker-based processing
        print("Step 2: Running workers for candidate retrieval and processing...")
        self.worker_manager.run_workers(self.feature)

        # Step 3: ML pipeline
        print("Step 3: Running ML pipeline...")
        self.ml_manager.run_ml_pipeline(self.feature)

        # Step 4: Output generation
        print("Step 4: Generating output...")
        extracted_rows = self.output_manager.save_output()

        print("Alligator entity linking pipeline completed successfully!")
        return extracted_rows

    def close_connections(self):
        """Cleanup resources and close connections."""
        from alligator.mongo import MongoConnectionManager

        try:
            MongoConnectionManager.close_connection()
        except Exception:
            pass
        print("Connections closed.")
