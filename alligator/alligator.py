"""
Alligator entity linking system - refactored with improved architecture.

This module provides the main Alligator class, now serving as a facade that delegates
work to the AlligatorCoordinator and its specialized managers while maintaining
backward compatibility with the existing API.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import aiohttp
import pandas as pd

from alligator.config import AlligatorConfig
from alligator.coordinator import AlligatorCoordinator
from alligator.fetchers import CandidateFetcher, LiteralFetcher, ObjectFetcher
from alligator.processors import RowBatchProcessor
from alligator.types import ColType


class Alligator:
    """
    Alligator entity linking system - refactored with improved architecture.

    This class now serves as a facade that delegates work to the AlligatorCoordinator
    and its specialized managers, while maintaining backward compatibility with the
    existing API.
    """

    def __init__(
        self,
        input_csv: str | Path | pd.DataFrame | None = None,
        output_csv: str | Path | None = None,
        dataset_name: str | None = None,
        table_name: str | None = None,
        target_rows: List[str] | None = None,
        target_columns: ColType | None = None,
        column_types: Mapping[str, Union[str, List[str]]] | None = None,
        worker_batch_size: int = 16,
        num_workers: Optional[int] = 1,
        max_candidates_in_result: int = 50,
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
        http_session_limit: int = 32,
        http_session_ssl_verify: bool = False,
        **kwargs,
    ) -> None:
        """Initialize Alligator with the new architecture."""

        # Create configuration object
        self.config = AlligatorConfig(
            input_csv=input_csv,
            output_csv=output_csv,
            dataset_name=dataset_name,
            table_name=table_name,
            target_rows=target_rows,
            target_columns=target_columns,
            column_types=column_types,
            worker_batch_size=worker_batch_size,
            num_workers=num_workers,
            max_candidates_in_result=max_candidates_in_result,
            entity_retrieval_endpoint=entity_retrieval_endpoint,
            entity_retrieval_token=entity_retrieval_token,
            object_retrieval_endpoint=object_retrieval_endpoint,
            literal_retrieval_endpoint=literal_retrieval_endpoint,
            selected_features=selected_features,
            candidate_retrieval_limit=candidate_retrieval_limit,
            ranker_model_path=ranker_model_path,
            reranker_model_path=reranker_model_path,
            ml_worker_batch_size=ml_worker_batch_size,
            num_ml_workers=num_ml_workers,
            top_n_cta_cpa_freq=top_n_cta_cpa_freq,
            doc_percentage_type_features=doc_percentage_type_features,
            save_output=save_output,
            save_output_to_csv=save_output_to_csv,
            correct_qids=correct_qids,
            http_session_limit=http_session_limit,
            http_session_ssl_verify=http_session_ssl_verify,
            **kwargs,
        )

        # Create coordinator
        self.coordinator = AlligatorCoordinator(self.config)

        # Expose key properties for backward compatibility
        self.input_csv = self.config.data.input_csv
        self.output_csv = self.config.data.output_csv
        self.dataset_name = self.config.data.dataset_name
        self.table_name = self.config.data.table_name
        self.target_rows = self.config.data.target_rows
        self.target_columns = self.config.data.target_columns
        self.worker_batch_size = self.config.worker.worker_batch_size
        self.num_workers = self.config.worker.num_workers
        self.max_candidates_in_result = self.config.retrieval.max_candidates_in_result
        self.entity_retrieval_endpoint = self.config.retrieval.entity_retrieval_endpoint
        self.entity_retrieval_token = self.config.retrieval.entity_retrieval_token
        self.object_retrieval_endpoint = self.config.retrieval.object_retrieval_endpoint
        self.literal_retrieval_endpoint = self.config.retrieval.literal_retrieval_endpoint
        self.candidate_retrieval_limit = self.config.retrieval.candidate_retrieval_limit
        self.ranker_model_path = self.config.ml.ranker_model_path
        self.reranker_model_path = self.config.ml.reranker_model_path
        self.ml_worker_batch_size = self.config.ml.ml_worker_batch_size
        self.num_ml_workers = self.config.ml.num_ml_workers
        self.top_n_cta_cpa_freq = self.config.feature.top_n_cta_cpa_freq
        self.doc_percentage_type_features = self.config.feature.doc_percentage_type_features
        self.correct_qids = self.config.data.correct_qids

        # Expose some internal properties for compatibility
        self._save_output = self.config.data.save_output
        self._save_output_to_csv = self.config.data.save_output_to_csv
        self._dry_run = self.config.data.dry_run
        self._mongo_uri = self.config.database.mongo_uri
        self._db_name = self.config.database.db_name
        self._DB_NAME = self.config.database.db_name
        self._INPUT_COLLECTION = self.config.database.input_collection
        self._ERROR_LOG_COLLECTION = self.config.database.error_log_collection
        self._CACHE_COLLECTION = self.config.database.cache_collection
        self._OBJECT_CACHE_COLLECTION = self.config.database.object_cache_collection
        self._LITERAL_CACHE_COLLECTION = self.config.database.literal_cache_collection
        self._http_session_limit = self.config.retrieval.http_session_limit
        self._http_session_ssl_verify = self.config.retrieval.http_session_ssl_verify

        # Delegate initialization to coordinator components
        self.mongo_wrapper = self.coordinator.data_manager.mongo_wrapper
        self.feature = self.coordinator.feature

        # Initialize fetchers to None for compatibility (they are created in coordinator)
        self.candidate_fetcher: Optional[CandidateFetcher] = None
        self.object_fetcher: Optional[ObjectFetcher] = None
        self.literal_fetcher: Optional[LiteralFetcher] = None
        self.row_processor: Optional[RowBatchProcessor] = None

    def run(self) -> List[Dict[str, Any]]:
        """Execute the entity linking pipeline using the coordinator."""
        return self.coordinator.run()

    def close_mongo_connection(self):
        """Cleanup when instance is destroyed."""
        self.coordinator.close_connections()

    def onboard_data(
        self,
        dataset_name: str | None = None,
        table_name: str | None = None,
        target_columns: ColType | None = None,
    ):
        """Delegate data onboarding to the data manager."""
        # Update config if parameters are provided
        if dataset_name is not None:
            self.config.data.dataset_name = dataset_name
        if table_name is not None:
            self.config.data.table_name = table_name
        if target_columns is not None:
            self.config.data.target_columns = target_columns

        # Perform onboarding
        self.coordinator.data_manager.onboard_data()

    def save_output(self) -> List[Dict[str, Any]]:
        """Delegate output saving to the output manager."""
        return self.coordinator.output_manager.save_output()

    # Backward compatibility methods (delegated to managers)
    async def _initialize_async_components(self) -> aiohttp.ClientSession:
        """Initialize async components (for backward compatibility)."""
        return await self.coordinator.worker_manager.initialize_async_components()

    def ml_worker(self, rank: int, stage: str, global_frequencies: Tuple):
        """ML worker method (for backward compatibility)."""
        return self.coordinator.ml_manager._ml_worker(rank, stage, global_frequencies)

    async def process_batch(self, docs):
        """Process batch method (for backward compatibility)."""
        # This would need the row processor, which is created in the worker
        # For now, we'll delegate to the coordinator's implementation

    async def worker_async(self, rank: int):
        """Async worker method (for backward compatibility)."""
        await self.coordinator.worker_manager._worker_async(rank, self.feature)

    def worker(self, rank: int):
        """Worker method (for backward compatibility)."""
        asyncio.run(self.worker_async(rank))

    def _extract_row_data(self, doc, header):
        """Extract row data (for backward compatibility)."""
        return self.coordinator.output_manager._extract_row_data(doc, header)
