import asyncio
import multiprocessing as mp
from typing import Dict, List

import aiohttp

from alligator import TIMEOUT
from alligator.config import AlligatorConfig
from alligator.database import DatabaseAccessMixin
from alligator.feature import Feature
from alligator.fetchers import CandidateFetcher, LiteralFetcher, ObjectFetcher
from alligator.log import get_logger
from alligator.mongo import MongoWrapper
from alligator.processors import RowBatchProcessor


class WorkerManager(DatabaseAccessMixin):
    """Manages worker processes for batch processing."""

    def __init__(self, config: AlligatorConfig):
        self.config = config
        self.logger = get_logger("worker_manager")
        self._mongo_uri = config.database.mongo_uri or "mongodb://gator-mongodb:27017/"
        self._db_name = config.database.db_name or "alligator_db"

    async def initialize_async_components(self) -> aiohttp.ClientSession:
        """Initialize components that require an active event loop."""
        connector = aiohttp.TCPConnector(
            limit=self.config.retrieval.http_session_limit,
            ssl=self.config.retrieval.http_session_ssl_verify,
        )
        session = aiohttp.ClientSession(connector=connector, timeout=TIMEOUT)
        self.logger.info(
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
            column_types=self.config.data.column_types,
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
        self.logger.info(f"Found {total_rows} tasks to process.")

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
        self.logger.info(f"Worker {rank} started.")
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
        num_workers = self.config.worker.num_workers or 1
        docs_per_worker = total_docs // num_workers
        remainder = total_docs % num_workers

        # Adjust batch size for this specific worker
        # Give extra documents to earlier workers if not evenly divisible
        if rank < remainder:
            batch_size = docs_per_worker + 1
            skip = rank * (docs_per_worker + 1)
        else:
            batch_size = docs_per_worker
            skip = (remainder * (docs_per_worker + 1)) + ((rank - remainder) * docs_per_worker)
        if batch_size <= 0:
            self.logger.warning(
                f"Worker {rank} has no documents to process (batch size is {batch_size})."
            )
            await session.close()
            return
        self.logger.info(f"Worker {rank} started with batch size {batch_size}, skipping {skip}.")

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
                self.logger.info(f"Worker {rank} processing batch of {len(todo_docs)} documents.")
                await self._process_batch(todo_docs, row_processor)
                self.logger.info(f"Worker {rank} processed {len(todo_docs)} documents.")
                todo_docs = [doc]
        if todo_docs:
            self.logger.info(
                f"Worker {rank} processing final batch of {len(todo_docs)} documents."
            )
            await self._process_batch(todo_docs, row_processor)
            self.logger.info(f"Worker {rank} finished processing documents.")

        await session.close()

    async def _process_batch(self, docs: List[Dict], row_processor: RowBatchProcessor):
        """Process a batch of documents."""
        if row_processor:
            await row_processor.process_rows_batch(docs)
