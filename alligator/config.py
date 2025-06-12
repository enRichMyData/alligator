"""
Configuration management for the Alligator entity linking system.

This module provides a structured approach to configuration management,
replacing the monolithic parameter handling in the main Alligator class.
"""

import multiprocessing as mp
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from alligator import PROJECT_ROOT
from alligator.typing import ColType


@dataclass
class DataConfig:
    """Configuration for data input/output and processing."""

    input_csv: Union[str, Path, pd.DataFrame, None] = None
    output_csv: Union[str, Path, None] = None
    dataset_name: Optional[str] = None
    table_name: Optional[str] = None
    target_rows: List[str] = field(default_factory=list)
    target_columns: Optional[ColType] = None
    save_output: bool = True
    save_output_to_csv: bool = True
    correct_qids: Dict[str, Union[str, List[str]]] = field(default_factory=dict)
    dry_run: bool = False
    testing_mode: bool = False  # New field for testing

    def __post_init__(self):
        """Validate and process data configuration after initialization."""
        self._validate_input()
        self._process_defaults()
        self._process_target_rows()
        self._process_correct_qids()

    def _validate_input(self):
        """Validate input data configuration."""
        if self.input_csv is None:
            raise ValueError("Input CSV or DataFrame must be provided.")

        if not isinstance(self.input_csv, (str, Path, pd.DataFrame)):
            raise ValueError("Input must be a file path (str or Path) or a pandas DataFrame.")

        if isinstance(self.input_csv, (str, Path)) and not os.path.exists(self.input_csv):
            raise FileNotFoundError(f"Input file '{self.input_csv}' does not exist.")

        # Validate output requirements - only require output name if we're actually saving to CSV
        if (
            self.save_output
            and self.save_output_to_csv
            and self.output_csv is None
            and isinstance(self.input_csv, pd.DataFrame)
        ):
            raise ValueError(
                "An output name must be specified if the input is a `pd.DataFrame` "
                "and save_output_to_csv is True"
            )

    def _process_defaults(self):
        """Process default values for dataset/table names and output path."""
        # Set default dataset name
        if self.dataset_name is None:
            self.dataset_name = uuid.uuid4().hex

        # Set default table name
        if self.table_name is None:
            if isinstance(self.input_csv, str):
                self.table_name = os.path.basename(self.input_csv).split(".")[0]
            else:
                self.table_name = "default_table"

        # Set default output path
        if (
            self.save_output
            and self.output_csv is None
            and self.save_output_to_csv
            and isinstance(self.input_csv, (str, Path))
        ):
            self.output_csv = os.path.splitext(self.input_csv)[0] + "_output.csv"

    def _process_target_rows(self):
        """Process and normalize target rows."""
        if self.target_rows:
            self.target_rows = [str(row) for row in self.target_rows]
            self.target_rows = list(set(self.target_rows))

    def _process_correct_qids(self):
        """Process and validate correct QIDs."""
        for key, value in self.correct_qids.items():
            if isinstance(value, str):
                self.correct_qids[key] = [value]
            elif not isinstance(value, list):
                raise ValueError(f"Correct QIDs for {key} must be a string or a list of strings.")


@dataclass
class WorkerConfig:
    """Configuration for worker processes and batch processing."""

    worker_batch_size: int = 64
    num_workers: Optional[int] = None

    def __post_init__(self):
        """Set default number of workers if not specified."""
        if self.num_workers is None:
            self.num_workers = max(1, mp.cpu_count() // 2)


@dataclass
class RetrievalConfig:
    """Configuration for entity and data retrieval endpoints."""

    entity_retrieval_endpoint: Optional[str] = None
    entity_retrieval_token: Optional[str] = None
    object_retrieval_endpoint: Optional[str] = None
    literal_retrieval_endpoint: Optional[str] = None
    candidate_retrieval_limit: int = 16
    max_candidates_in_result: int = 5
    http_session_limit: int = 32
    http_session_ssl_verify: bool = False

    def __post_init__(self):
        """Validate and process retrieval configuration."""
        # Get endpoints from environment if not provided
        self.entity_retrieval_endpoint = self.entity_retrieval_endpoint or os.getenv(
            "ENTITY_RETRIEVAL_ENDPOINT"
        )
        self.entity_retrieval_token = self.entity_retrieval_token or os.getenv(
            "ENTITY_RETRIEVAL_TOKEN"
        )
        self.object_retrieval_endpoint = self.object_retrieval_endpoint or os.getenv(
            "OBJECT_RETRIEVAL_ENDPOINT"
        )
        self.literal_retrieval_endpoint = self.literal_retrieval_endpoint or os.getenv(
            "LITERAL_RETRIEVAL_ENDPOINT"
        )

        # Validate required endpoints
        if not self.entity_retrieval_endpoint:
            raise ValueError("Entity retrieval endpoint must be provided.")
        if not self.entity_retrieval_token:
            raise ValueError("Entity retrieval token must be provided.")


@dataclass
class MLConfig:
    """Configuration for machine learning models and processing."""

    ranker_model_path: Optional[str] = None
    reranker_model_path: Optional[str] = None
    ml_worker_batch_size: int = 256
    num_ml_workers: int = 2
    selected_features: Optional[List[str]] = None

    def __post_init__(self):
        """Set default model paths if not specified."""
        if self.ranker_model_path is None:
            self.ranker_model_path = os.path.join(PROJECT_ROOT, "alligator", "models", "ranker.h5")

        if self.reranker_model_path is None:
            self.reranker_model_path = os.path.join(
                PROJECT_ROOT, "alligator", "models", "reranker.h5"
            )


@dataclass
class FeatureConfig:
    """Configuration for feature computation and processing."""

    top_n_cta_cpa_freq: int = 3
    doc_percentage_type_features: float = 1.0

    def __post_init__(self):
        """Validate feature configuration."""
        if not (0 < self.doc_percentage_type_features <= 1):
            raise ValueError("doc_percentage_type_features must be between 0 and 1 (exclusive).")


@dataclass
class DatabaseConfig:
    """Configuration for database connections and collections."""

    mongo_uri: Optional[str] = None
    db_name: Optional[str] = None
    input_collection: str = "input_data"
    error_log_collection: str = "error_logs"
    cache_collection: str = "candidate_cache"
    object_cache_collection: str = "object_cache"
    literal_cache_collection: str = "literal_cache"

    # Default constants
    _DEFAULT_MONGO_URI = "mongodb://gator-mongodb:27017/"
    _DEFAULT_DB_NAME = "alligator_db"

    def __post_init__(self):
        """Set default database configuration."""
        if self.mongo_uri is None:
            self.mongo_uri = self._DEFAULT_MONGO_URI
        if self.db_name is None:
            self.db_name = self._DEFAULT_DB_NAME


class AlligatorConfig:
    """
    Centralized configuration management for the Alligator entity linking system.

    This class aggregates all configuration sections and provides a single
    interface for configuration management with validation and defaults.
    """

    def __init__(
        self,
        # Data configuration
        input_csv: Union[str, Path, pd.DataFrame, None] = None,
        output_csv: Union[str, Path, None] = None,
        dataset_name: Optional[str] = None,
        table_name: Optional[str] = None,
        target_rows: Optional[List[str]] = None,
        target_columns: Optional[ColType] = None,
        save_output: bool = True,
        save_output_to_csv: bool = True,
        correct_qids: Optional[Dict[str, Union[str, List[str]]]] = None,
        dry_run: bool = False,
        # Worker configuration
        worker_batch_size: int = 64,
        num_workers: Optional[int] = None,
        # Retrieval configuration
        entity_retrieval_endpoint: Optional[str] = None,
        entity_retrieval_token: Optional[str] = None,
        object_retrieval_endpoint: Optional[str] = None,
        literal_retrieval_endpoint: Optional[str] = None,
        candidate_retrieval_limit: int = 16,
        max_candidates_in_result: int = 5,
        http_session_limit: int = 32,
        http_session_ssl_verify: bool = False,
        # ML configuration
        ranker_model_path: Optional[str] = None,
        reranker_model_path: Optional[str] = None,
        ml_worker_batch_size: int = 256,
        num_ml_workers: int = 2,
        selected_features: Optional[List[str]] = None,
        # Feature configuration
        top_n_cta_cpa_freq: int = 3,
        doc_percentage_type_features: float = 1.0,
        # Database configuration
        mongo_uri: Optional[str] = None,
        db_name: Optional[str] = None,
        # Testing configuration
        testing_mode: bool = False,
        # Additional keyword arguments for future extensibility
        **kwargs,
    ):
        """Initialize the Alligator configuration with validation."""

        # Initialize configuration sections
        self.data = DataConfig(
            input_csv=input_csv,
            output_csv=output_csv,
            dataset_name=dataset_name,
            table_name=table_name,
            target_rows=target_rows or [],
            target_columns=target_columns,
            save_output=save_output,
            save_output_to_csv=save_output_to_csv,
            correct_qids=correct_qids or {},
            dry_run=dry_run,
            testing_mode=testing_mode,
        )

        self.worker = WorkerConfig(
            worker_batch_size=worker_batch_size,
            num_workers=num_workers,
        )

        self.retrieval = RetrievalConfig(
            entity_retrieval_endpoint=entity_retrieval_endpoint,
            entity_retrieval_token=entity_retrieval_token,
            object_retrieval_endpoint=object_retrieval_endpoint,
            literal_retrieval_endpoint=literal_retrieval_endpoint,
            candidate_retrieval_limit=candidate_retrieval_limit,
            max_candidates_in_result=max_candidates_in_result,
            http_session_limit=http_session_limit,
            http_session_ssl_verify=http_session_ssl_verify,
        )

        self.ml = MLConfig(
            ranker_model_path=ranker_model_path,
            reranker_model_path=reranker_model_path,
            ml_worker_batch_size=ml_worker_batch_size,
            num_ml_workers=num_ml_workers,
            selected_features=selected_features,
        )

        self.feature = FeatureConfig(
            top_n_cta_cpa_freq=top_n_cta_cpa_freq,
            doc_percentage_type_features=doc_percentage_type_features,
        )

        self.database = DatabaseConfig(
            mongo_uri=kwargs.get("mongo_uri", mongo_uri),
            db_name=kwargs.get("db_name", db_name),
        )

        # Store any additional kwargs for extensibility
        self._extra_kwargs = {k: v for k, v in kwargs.items() if k not in ["mongo_uri", "db_name"]}

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        config_dict = {}

        for section_name in ["data", "worker", "retrieval", "ml", "feature", "database"]:
            section = getattr(self, section_name)
            if hasattr(section, "__dict__"):
                config_dict[section_name] = section.__dict__.copy()

        config_dict["extra"] = self._extra_kwargs
        return config_dict

    def validate(self) -> bool:
        """Validate the complete configuration."""
        try:
            # Configuration sections are validated in their __post_init__ methods
            # This method can be extended for cross-section validation
            return True
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")

    def __repr__(self) -> str:
        """String representation of the configuration."""
        return (
            f"AlligatorConfig(data={self.data}, "
            f"worker={self.worker}, "
            f"retrieval={self.retrieval}, "
            f"ml={self.ml}, "
            f"feature={self.feature}, "
            f"database={self.database})"
        )
