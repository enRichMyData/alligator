import asyncio
import multiprocessing as mp
import os
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Counter, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
from column_classifier import ColumnClassifier

from alligator import PROJECT_ROOT, TIMEOUT
from alligator.database import DatabaseAccessMixin
from alligator.feature import Feature
from alligator.fetchers import CandidateFetcher, LiteralFetcher, ObjectFetcher
from alligator.ml import MLWorker
from alligator.mongo import MongoWrapper
from alligator.processors import RowBatchProcessor
from alligator.typing import ColType


class Alligator(DatabaseAccessMixin):
    """
    Alligator entity linking system with hidden MongoDB configuration.
    """

    _DEFAULT_MONGO_URI = "mongodb://gator-mongodb:27017/"  # Change this to a class-level default
    _DB_NAME = "alligator_db"
    _INPUT_COLLECTION = "input_data"
    _ERROR_LOG_COLLECTION = "error_logs"
    _CACHE_COLLECTION = "candidate_cache"
    _OBJECT_CACHE_COLLECTION = "object_cache"
    _LITERAL_CACHE_COLLECTION = "literal_cache"

    def __init__(
        self,
        input_csv: str | Path | pd.DataFrame | None = None,
        output_csv: str | Path | None = None,
        dataset_name: str | None = None,
        table_name: str | None = None,
        target_rows: List[str] | None = None,
        target_columns: ColType | None = None,
        worker_batch_size: int = 64,
        num_workers: Optional[int] = None,
        max_candidates_in_result: int = 5,
        entity_retrieval_endpoint: Optional[str] = None,
        entity_retrieval_token: Optional[str] = None,
        object_retrieval_endpoint: Optional[str] = None,
        literal_retrieval_endpoint: Optional[str] = None,
        selected_features: Optional[List[str]] = None,
        candidate_retrieval_limit: int = 16,
        ranker_model_path: Optional[str] = None,
        reranker_model_path: Optional[str] = None,
        ml_worker_batch_size: int = 256,
        num_ml_workers: int = 2,
        top_n_cta_cpa_freq: int = 3,
        doc_percentage_type_features: float = 1.0,
        save_output: bool = True,
        save_output_to_csv: bool = True,
        correct_qids: Dict[str, str | List[str]] | None = None,
        http_session_limit: int = 50,
        http_session_ssl_verify: bool = False,
        **kwargs,
    ) -> None:
        # Checks
        if input_csv is None:
            raise ValueError("Input CSV or DataFrame must be provided.")
        self.input_csv = input_csv
        self.output_csv = output_csv
        if save_output and self.output_csv is None and save_output_to_csv:
            if isinstance(self.input_csv, pd.DataFrame):
                raise ValueError(
                    "An output name must be specified is the input is a `pd.Dataframe`"
                )
            self.output_csv = os.path.splitext(input_csv)[0] + "_output.csv"
        if dataset_name is None:
            dataset_name = uuid.uuid4().hex
        if table_name is None:
            table_name = os.path.basename(self.input_csv).split(".")[0]
        self.dataset_name = dataset_name
        self.table_name = table_name

        # Set up rows and columns
        self.target_rows = target_rows or []
        self.target_rows = [str(row) for row in self.target_rows]
        self.target_rows = list(set(self.target_rows))
        self.target_columns = target_columns

        # Worker config
        self.worker_batch_size = worker_batch_size
        self.num_workers = num_workers or max(1, mp.cpu_count() // 2)

        # Retrievers config
        self.max_candidates_in_result = max_candidates_in_result
        self.entity_retrieval_endpoint = entity_retrieval_endpoint or os.getenv(
            "ENTITY_RETRIEVAL_ENDPOINT"
        )
        if not self.entity_retrieval_endpoint:
            raise ValueError("Entity retrieval endpoint must be provided.")
        self.entity_retrieval_token = entity_retrieval_token or os.getenv("ENTITY_RETRIEVAL_TOKEN")
        if not self.entity_retrieval_token:
            raise ValueError("Entity retrieval token must be provided.")
        self.object_retrieval_endpoint = object_retrieval_endpoint or os.getenv(
            "OBJECT_RETRIEVAL_ENDPOINT"
        )
        self.literal_retrieval_endpoint = literal_retrieval_endpoint or os.getenv(
            "LITERAL_RETRIEVAL_ENDPOINT"
        )
        self.candidate_retrieval_limit = candidate_retrieval_limit

        # ML config
        self.ranker_model_path = ranker_model_path or os.path.join(
            PROJECT_ROOT, "alligator", "models", "ranker.h5"
        )
        self.reranker_model_path = reranker_model_path or os.path.join(
            PROJECT_ROOT, "alligator", "models", "reranker.h5"
        )
        self.ml_worker_batch_size = ml_worker_batch_size
        self.num_ml_workers = num_ml_workers

        # Feature config
        self.top_n_cta_cpa_freq = top_n_cta_cpa_freq
        self.doc_percentage_type_features = doc_percentage_type_features
        if not (0 < self.doc_percentage_type_features <= 1):
            raise ValueError("doc_percentage_type_features must be between 0 and 1 (exclusive).")
        self._http_session_limit = http_session_limit
        self._http_session_ssl_verify = http_session_ssl_verify
        self._shared_http_session: Optional[aiohttp.ClientSession] = None
        self._mongo_uri = kwargs.pop("mongo_uri", None) or self._DEFAULT_MONGO_URI
        self._db_name = kwargs.pop("db_name", None) or self._DB_NAME
        self._save_output = save_output
        self._save_output_to_csv = save_output_to_csv
        self.correct_qids = correct_qids or {}
        for key, value in self.correct_qids.items():
            if isinstance(value, str):
                self.correct_qids[key] = [value]
            elif not isinstance(value, list):
                raise ValueError(f"Correct QIDs for {key} must be a string or a list of strings.")
        self._dry_run = kwargs.pop("dry_run", False)

        # Initializations
        self.mongo_wrapper = MongoWrapper(
            self._mongo_uri,
            self._DB_NAME,
            self._INPUT_COLLECTION,
            self._ERROR_LOG_COLLECTION,
        )
        self.feature = Feature(
            dataset_name,
            table_name,
            top_n_cta_cpa_freq=top_n_cta_cpa_freq,
            features=selected_features,
            db_name=self._DB_NAME,
            mongo_uri=self._mongo_uri,
            input_collection=self._INPUT_COLLECTION,
        )

        # Initialize fetchers to None; they will be created in _initialize_async_components
        self.candidate_fetcher: Optional[CandidateFetcher] = None
        self.object_fetcher: Optional[ObjectFetcher] = None
        self.literal_fetcher: Optional[LiteralFetcher] = None
        self.row_processor: Optional[RowBatchProcessor] = None

        # Create indexes
        self.mongo_wrapper.create_indexes()

        # Onboard data
        self.onboard_data(self.dataset_name, self.table_name, target_columns=self.target_columns)

    async def _initialize_async_components(self):
        """Initializes components that require an active event loop or async operations.
        Each process must create its own HTTP session!
        """
        connector = aiohttp.TCPConnector(
            limit=self._http_session_limit, ssl=self._http_session_ssl_verify
        )
        session = aiohttp.ClientSession(connector=connector, timeout=TIMEOUT)
        print(
            f"Created HTTP session with limit {self._http_session_limit} "
            f"and SSL verify {self._http_session_ssl_verify}."
        )

        self.candidate_fetcher = CandidateFetcher(
            self.entity_retrieval_endpoint,
            self.entity_retrieval_token,
            self.candidate_retrieval_limit,
            self.feature,
            session=session,
            db_name=self._DB_NAME,
            mongo_uri=self._mongo_uri,
            input_collection=self._INPUT_COLLECTION,
            cache_collection=self._CACHE_COLLECTION,
        )
        if self.object_retrieval_endpoint:
            self.object_fetcher = ObjectFetcher(
                self.object_retrieval_endpoint,
                self.entity_retrieval_token,
                session=session,
                db_name=self._DB_NAME,
                mongo_uri=self._mongo_uri,
                cache_collection=self._OBJECT_CACHE_COLLECTION,
            )

        if self.literal_retrieval_endpoint:
            self.literal_fetcher = LiteralFetcher(
                self.literal_retrieval_endpoint,
                self.entity_retrieval_token,
                session=session,
                db_name=self._DB_NAME,
                mongo_uri=self._mongo_uri,
                cache_collection=self._LITERAL_CACHE_COLLECTION,
            )
        self.row_processor = RowBatchProcessor(
            self.dataset_name,
            self.table_name,
            self.feature,
            self.candidate_fetcher,
            self.object_fetcher,
            self.literal_fetcher,
            self.max_candidates_in_result,
            db_name=self._DB_NAME,
            mongo_uri=self._mongo_uri,
            input_collection=self._INPUT_COLLECTION,
        )
        self._shared_http_session = session  # For cleanup in this process

    async def close(self):
        """Closes resources like the HTTP session."""
        if (
            hasattr(self, "_shared_http_session")
            and self._shared_http_session
            and not self._shared_http_session.closed
        ):
            await self._shared_http_session.close()
            print("HTTP session closed.")

    def close_mongo_connection(self):
        """Cleanup when instance is destroyed"""
        from alligator.mongo import MongoConnectionManager

        try:
            MongoConnectionManager.close_connection()
        except Exception:
            pass

    def onboard_data(
        self,
        dataset_name: str | None = None,
        table_name: str | None = None,
        target_columns: ColType | None = None,
    ):
        """Efficiently load data into MongoDB using batched inserts."""
        start_time = time.perf_counter()

        # Get database connection
        db = self.get_db()
        input_collection = db["input_data"]

        # Step 1: Determine data source and extract sample for classification
        if isinstance(self.input_csv, pd.DataFrame):
            df = self.input_csv
            sample = df
            total_rows = len(df)
            is_csv_path = False
        else:
            sample = pd.read_csv(self.input_csv, nrows=32)
            total_rows = -1
            is_csv_path = True

        print(f"Onboarding {total_rows} rows for dataset '{dataset_name}', table '{table_name}'")

        # Step 2: Perform column classification
        classifier = ColumnClassifier(model_type="fast")
        classification_results = classifier.classify_multiple_tables([sample])
        table_classification = classification_results[0].get("table_1", {})

        ne_cols, lit_cols, ignored_cols = {}, {}, []
        ne_types = {"PERSON", "OTHER", "ORGANIZATION", "LOCATION"}
        lit_types = {"NUMBER", "DATE", "STRING"}

        classified_columns = {}
        for idx, col in enumerate(sample.columns):
            col_result = table_classification.get(col, {})
            classification = col_result.get("classification", "UNKNOWN")
            if classification in ne_types:
                ne_cols[str(idx)] = classification
            elif classification in lit_types:
                if classification == "DATE":
                    classification = "DATETIME"
                lit_cols[str(idx)] = classification
            classified_columns[str(idx)] = classification

        if target_columns is not None:
            ne_cols = target_columns.get("NE", ne_cols)
            for col in ne_cols:
                ne_cols[col] = classified_columns.get(col, "UNKNOWN")
            lit_cols = target_columns.get("LIT", lit_cols)
            for col in lit_cols:
                if not lit_cols[col]:
                    lit_cols[col] = classified_columns.get(col, "UNKNOWN")
            ignored_cols = target_columns.get("IGNORED", ignored_cols)

        all_recognized_cols = set(ne_cols.keys()) | set(lit_cols.keys())
        all_cols = set([str(i) for i in range(len(sample.columns))])
        if len(all_recognized_cols) != len(all_cols):
            ignored_cols.extend(list(all_cols - all_recognized_cols))
        ignored_cols = list(set(ignored_cols))
        context_cols = list(set([str(i) for i in range(len(sample.columns))]) - set(ignored_cols))
        context_cols = sorted(context_cols, key=lambda x: int(x))

        # Step 3: Define a chunk generator function
        def get_chunks():
            """Generator that yields chunks of rows, handling both DF and CSV."""
            if self._dry_run:
                if is_csv_path:
                    yield pd.read_csv(self.input_csv, nrows=1), 0
                else:
                    yield df.iloc[:1], 0
            else:
                if is_csv_path:
                    chunk_size = 2048
                    row_count = 0
                    for chunk in pd.read_csv(self.input_csv, chunksize=chunk_size):
                        yield chunk, row_count
                        row_count += len(chunk)
                else:
                    chunk_size = (
                        1024 if total_rows > 100000 else 2048 if total_rows > 10000 else 4096
                    )
                    total_chunks = (total_rows + chunk_size - 1) // chunk_size
                    for chunk_idx in range(total_chunks):
                        chunk_start = chunk_idx * chunk_size
                        chunk_end = min(chunk_start + chunk_size, total_rows)
                        yield df.iloc[chunk_start:chunk_end], chunk_start

        # Step 4: Process all chunks using the generator
        processed_rows = 0
        chunk_idx = 0

        for chunk, start_idx in get_chunks():
            chunk_idx += 1
            documents = []
            for i, (_, row) in enumerate(chunk.iterrows()):
                row_id = start_idx + i
                if str(row_id) not in self.target_rows and self.target_rows:
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
                correct_qids = {}
                for col_id, _ in ne_cols.items():
                    key = f"{row_id}-{col_id}"
                    if key in self.correct_qids:
                        correct_qids[key] = self.correct_qids[key]
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
                            f"Chunk {chunk_idx}: "
                            f"Processed {chunk_size} rows (total: {processed_rows}) "
                            f"({rows_per_second:.1f} rows/sec)"
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

        total_time = time.perf_counter() - start_time
        print(f"Data onboarding complete for dataset '{dataset_name}' and table '{table_name}'")
        print(
            f"Onboarded {processed_rows} rows in {total_time:.2f} seconds "
            f"({processed_rows/total_time:.1f} rows/sec)"
        )

    def save_output(self):
        """Retrieves processed documents from MongoDB using memory-efficient streaming.

        For large tables, this avoids loading all results into memory at once.
        Instead, it processes documents in batches and writes directly to CSV.
        """
        db = self.get_db()
        input_collection = db[self._INPUT_COLLECTION]

        # Determine if we need to write to CSV or return full results
        stream_to_csv = self._save_output_to_csv and isinstance(self.output_csv, (str, Path))

        # Get header information
        header = None
        if isinstance(self.input_csv, pd.DataFrame):
            header = self.input_csv.columns.tolist()
        elif isinstance(self.input_csv, str):
            header = pd.read_csv(self.input_csv, nrows=0).columns.tolist()

        # Get first document to determine column count if header is still None
        sample_doc = input_collection.find_one(
            {"dataset_name": self.dataset_name, "table_name": self.table_name}
        )
        if not sample_doc:
            print("No documents found for the specified dataset and table.")
            return []

        if header is None:
            print("Could not extract header from input table, using generic column names.")
            header = [f"col_{i}" for i in range(len(sample_doc["data"]))]

        # Process in batches with cursor
        batch_size = 1024  # Process 1024 documents at a time

        # Only fetch fields we actually need to reduce network transfer
        projection = {"data": 1, "el_results": 1, "classified_columns.NE": 1}
        cursor = input_collection.find(
            {"dataset_name": self.dataset_name, "table_name": self.table_name},
            projection=projection,
        ).batch_size(batch_size)

        # Count documents if streaming to CSV for progress reporting
        if stream_to_csv:
            total_docs = input_collection.count_documents(
                {"dataset_name": self.dataset_name, "table_name": self.table_name}
            )
            print(f"Streaming {total_docs} documents to CSV...")

        # Setup for handling results
        all_rows = [] if not stream_to_csv else None
        current_chunk = []
        processed_count = 0
        chunk_size = 256  # Size for chunked CSV writing

        # Process all documents
        for doc in cursor:
            # Process each document - this is the common code between both paths
            row_data = self._extract_row_data(doc, header)

            if stream_to_csv:
                # For CSV streaming, collect into chunk
                current_chunk.append(row_data)
                processed_count += 1

                # Write chunk when it reaches desired size
                if len(current_chunk) >= chunk_size:
                    self._write_csv_chunk(current_chunk, processed_count, chunk_size, total_docs)
                    current_chunk = []
            else:
                # For memory collection, just append to results
                all_rows.append(row_data)
                processed_count += 1

                # Report progress periodically
                if processed_count % batch_size == 0:
                    print(f"Processed {processed_count} rows...")

        # Handle any remaining rows for CSV streaming
        if stream_to_csv and current_chunk:
            self._write_csv_chunk(current_chunk, processed_count, chunk_size, total_docs)
            print(f"Results saved to '{self.output_csv}'. Total rows: {processed_count}")
            return []
        elif not stream_to_csv:
            print(f"Retrieved {processed_count} rows total")
            return all_rows
        else:
            return []

    def _extract_row_data(self, doc, header):
        """Extract row data from a MongoDB document.

        Encapsulates the common logic for formatting a row from a document.
        """
        # Create base row data with original values
        row_data = dict(zip(header, doc["data"]))
        el_results = doc.get("el_results", {})

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

    def _write_csv_chunk(self, chunk, processed_count, chunk_size, total_docs):
        """Write a chunk of data to CSV.

        Encapsulates the CSV writing logic.
        """
        # Create DataFrame and append to CSV
        chunk_df = pd.DataFrame(chunk)

        # Use mode='a' (append) for all chunks after the first
        mode = "w" if processed_count <= len(chunk) else "a"
        # Only include header for the first chunk
        header_option = True if processed_count <= len(chunk) else False

        # Write chunk to CSV
        chunk_df.to_csv(self.output_csv, index=False, mode=mode, header=header_option)

        # Report progress
        print(f"Processed {processed_count}/{total_docs} rows...")

    @staticmethod
    def ml_worker(
        rank: int,
        stage: str,
        global_frequencies: Tuple[
            Dict[str, Counter] | None,
            Dict[str, Counter] | None,
            Dict[str, Dict[str, Counter]] | None,
        ],
        table_name: str,
        dataset_name: str,
        error_log_collection_name: str,
        input_collection_name: str,
        ranker_model_path: str,
        reranker_model_path: str,
        ml_worker_batch_size: int,
        max_candidates_rerank: int,
        top_n_cta_cpa_freq: int,
        selected_features: List[str],
        mongo_uri: str,
        db_name: str,
    ):
        """Static method wrapper for ML workers, suitable for multiprocessing."""
        model_path_to_use = ranker_model_path if stage == "rank" else reranker_model_path
        max_candidates_for_stage = -1 if stage == "rank" else max_candidates_rerank

        worker = MLWorker(
            rank,
            table_name=table_name,
            dataset_name=dataset_name,
            stage=stage,
            error_log_collection=error_log_collection_name,
            input_collection=input_collection_name,
            model_path=model_path_to_use,
            batch_size=ml_worker_batch_size,
            max_candidates_in_result=max_candidates_for_stage,
            top_n_cta_cpa_freq=top_n_cta_cpa_freq,
            features=selected_features,
            mongo_uri=mongo_uri,
            db_name=db_name,
        )
        return worker.run(global_frequencies=global_frequencies)

    async def process_batch(self, docs):
        tasks_by_table = {}
        for doc in docs:
            dataset_name = doc["dataset_name"]
            table_name = doc["table_name"]
            tasks_by_table.setdefault((dataset_name, table_name), []).append(doc)

        for (dataset_name, table_name), table_docs in tasks_by_table.items():
            await self.row_processor.process_rows_batch(table_docs)

    async def worker_async(self, rank: int):
        await self._initialize_async_components()

        db = self.get_db()
        input_collection = db[self._INPUT_COLLECTION]

        total_docs = input_collection.count_documents(
            {
                "dataset_name": self.dataset_name,
                "table_name": self.table_name,
            }
        )

        docs_per_worker = total_docs // self.num_workers
        remainder = total_docs % self.num_workers

        # Adjust batch size for this specific worker
        # Give extra documents to earlier workers if not evenly divisible
        if rank < remainder:
            batch_size = docs_per_worker + 1
            skip = rank * (docs_per_worker + 1)
        else:
            batch_size = docs_per_worker
            skip = (remainder * (docs_per_worker + 1)) + ((rank - remainder) * docs_per_worker)
        print(f"Worker {rank} started with batch size {batch_size}, skipping {skip}.")

        # Find documents to process for this worker using skip/limit
        cursor = (
            input_collection.find(
                {
                    "dataset_name": self.dataset_name,
                    "table_name": self.table_name,
                }
            )
            .skip(skip)
            .limit(batch_size)
        )

        todo_docs = []
        for doc in cursor:
            input_collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "DOING"}})
            if len(todo_docs) < self.worker_batch_size:
                todo_docs.append(doc)
            else:
                print(f"Worker {rank} processing batch of {len(todo_docs)} documents.")
                await self.process_batch(todo_docs)
                print(f"Worker {rank} processed {len(todo_docs)} documents.")
                todo_docs = [doc]
        if todo_docs:
            print(f"Worker {rank} processing final batch of {len(todo_docs)} documents.")
            await self.process_batch(todo_docs)
            print(f"Worker {rank} finished processing documents.")

        await self.close()

    def worker(self, rank: int):
        asyncio.run(self.worker_async(rank))

    def run(self):
        db = self.get_db()
        input_collection = db[self._INPUT_COLLECTION]

        total_rows = self.mongo_wrapper.count_documents(input_collection, {"status": "TODO"})
        print(f"Found {total_rows} tasks to process.")

        processes = []
        for rank in range(self.num_workers):
            p = mp.Process(target=self.worker, args=(rank,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            p.close()

        pool = mp.Pool(processes=self.num_workers)
        try:
            common_ml_args = {
                "table_name": self.table_name,
                "dataset_name": self.dataset_name,
                "error_log_collection_name": self._ERROR_LOG_COLLECTION,
                "input_collection_name": self._INPUT_COLLECTION,
                "ranker_model_path": self.ranker_model_path,
                "reranker_model_path": self.reranker_model_path,
                "ml_worker_batch_size": self.ml_worker_batch_size,
                "max_candidates_rerank": self.max_candidates_in_result,
                "top_n_cta_cpa_freq": self.top_n_cta_cpa_freq,
                "selected_features": self.feature.selected_features,
                "mongo_uri": self._mongo_uri,
                "db_name": self._DB_NAME,
            }

            # First ML ranking stage
            rank_stage_partial_func = partial(
                Alligator.ml_worker,  # Call the static method
                stage="rank",
                global_frequencies=(None, None, None),
                **common_ml_args,
            )
            pool.map(rank_stage_partial_func, range(self.num_ml_workers))
            print("ML Rank stage complete.")

            # Compute global frequencies (this happens in the main process)
            print("Computing global frequencies...")
            (
                type_frequencies,
                predicate_frequencies,
                predicate_pair_frequencies,
            ) = self.feature.compute_global_frequencies(
                docs_to_process=self.doc_percentage_type_features, random_sample=False
            )
            print("Global frequencies computed.")

            # Second ML ranking stage with global frequencies
            rerank_stage_partial_func = partial(
                Alligator.ml_worker,
                stage="rerank",
                global_frequencies=(
                    type_frequencies,
                    predicate_frequencies,
                    predicate_pair_frequencies,
                ),
                **common_ml_args,
            )
            pool.map(rerank_stage_partial_func, range(self.num_ml_workers))
            print("ML Rerank stage complete.")
        finally:
            pool.close()
            pool.join()

        print("All tasks have been processed.")
        if self._save_output:
            extracted_rows = self.save_output()
        else:
            extracted_rows = []
        return extracted_rows
