import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from alligator.config import (
    AlligatorConfig,
    DatabaseConfig,
    DataConfig,
    FeatureConfig,
    MLConfig,
    RetrievalConfig,
    WorkerConfig,
)


class TestDataConfig:
    def test_dataconfig_with_csv_file(self):
        """Test DataConfig with a CSV file path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\nval1,val2\n")
            temp_path = f.name

        try:
            config = DataConfig(input_csv=temp_path)
            assert str(config.input_csv) == temp_path
            assert config.dataset_name is not None
            assert config.table_name == Path(temp_path).stem
        finally:
            os.unlink(temp_path)

    def test_dataconfig_with_dataframe(self):
        """Test DataConfig with a pandas DataFrame."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        config = DataConfig(input_csv=df, output_csv="test_output.csv")
        assert isinstance(config.input_csv, pd.DataFrame)
        assert config.output_csv == "test_output.csv"

    def test_dataconfig_missing_output_for_dataframe(self):
        """Test that DataConfig raises error when DataFrame is provided without output."""
        df = pd.DataFrame({"col1": [1, 2]})
        with pytest.raises(ValueError, match="An output name must be specified"):
            DataConfig(input_csv=df, save_output=True, save_output_to_csv=True)

    def test_dataconfig_column_types_processing(self):
        """Test column types processing in DataConfig."""
        df = pd.DataFrame({"col1": [1, 2]})
        column_types = {"0": ["Q5"], "1": "Q11424", 2: ["Q515"]}  # Single string  # Non-string key

        config = DataConfig(input_csv=df, output_csv="test.csv", column_types=column_types)

        assert config.column_types["0"] == ["Q5"]
        assert config.column_types["1"] == ["Q11424"]
        assert config.column_types["2"] == ["Q515"]

    def test_dataconfig_invalid_column_types(self):
        """Test DataConfig with invalid column types."""
        df = pd.DataFrame({"col1": [1, 2]})
        with pytest.raises(ValueError, match="must be a string or list of strings"):
            DataConfig(
                input_csv=df,
                output_csv="test.csv",
                column_types={"0": 123},  # type: ignore # Invalid type for testing
            )

    def test_dataconfig_correct_qids_processing(self):
        """Test correct QIDs processing."""
        df = pd.DataFrame({"col1": [1, 2]})
        correct_qids = {"0-0": "Q12345", "0-1": ["Q67890", "Q11111"]}

        config = DataConfig(input_csv=df, output_csv="test.csv", correct_qids=correct_qids)

        assert config.correct_qids["0-0"] == ["Q12345"]
        assert config.correct_qids["0-1"] == ["Q67890", "Q11111"]


class TestWorkerConfig:
    def test_worker_config_defaults(self):
        """Test WorkerConfig with default values."""
        config = WorkerConfig()
        assert config.worker_batch_size == 64
        assert config.num_workers is not None
        assert config.num_workers > 0

    def test_worker_config_custom_values(self):
        """Test WorkerConfig with custom values."""
        config = WorkerConfig(worker_batch_size=128, num_workers=8)
        assert config.worker_batch_size == 128
        assert config.num_workers == 8


class TestRetrievalConfig:
    def test_retrieval_config_defaults(self):
        """Test RetrievalConfig with default values."""
        config = RetrievalConfig()
        assert config.candidate_retrieval_limit == 16
        assert config.max_candidates_in_result == 5
        assert config.http_session_limit == 32
        assert config.http_session_ssl_verify is False

    def test_retrieval_config_custom_values(self):
        """Test RetrievalConfig with custom values."""
        config = RetrievalConfig(
            entity_retrieval_endpoint="https://test.com",
            entity_retrieval_token="test_token",
            candidate_retrieval_limit=50,
        )
        assert config.entity_retrieval_endpoint == "https://test.com"
        assert config.entity_retrieval_token == "test_token"
        assert config.candidate_retrieval_limit == 50


class TestMLConfig:
    def test_ml_config_defaults(self):
        """Test MLConfig with default model paths."""
        config = MLConfig()
        assert config.ranker_model_path is not None
        assert "ranker.h5" in config.ranker_model_path
        assert config.reranker_model_path is not None
        assert "reranker.h5" in config.reranker_model_path
        assert config.ml_worker_batch_size == 256
        assert config.num_ml_workers == 2


class TestFeatureConfig:
    def test_feature_config_defaults(self):
        """Test FeatureConfig with default values."""
        config = FeatureConfig()
        assert config.top_n_cta_cpa_freq == 3
        assert config.doc_percentage_type_features == 1.0

    def test_feature_config_invalid_percentage(self):
        """Test FeatureConfig with invalid percentage value."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            FeatureConfig(doc_percentage_type_features=1.5)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            FeatureConfig(doc_percentage_type_features=0.0)


class TestDatabaseConfig:
    def test_database_config_defaults(self):
        """Test DatabaseConfig with default values."""
        config = DatabaseConfig()
        assert config.mongo_uri == "mongodb://gator-mongodb:27017/"
        assert config.db_name == "alligator_db"
        assert config.input_collection == "input_data"

    def test_database_config_custom_values(self):
        """Test DatabaseConfig with custom values."""
        config = DatabaseConfig(mongo_uri="mongodb://localhost:27017/", db_name="test_db")
        assert config.mongo_uri == "mongodb://localhost:27017/"
        assert config.db_name == "test_db"


class TestAlligatorConfig:
    def test_alligator_config_basic(self):
        """Test basic AlligatorConfig initialization."""
        df = pd.DataFrame({"col1": [1, 2]})
        config = AlligatorConfig(input_csv=df, output_csv="test.csv", dataset_name="test_dataset")

        assert isinstance(config.data.input_csv, pd.DataFrame)
        assert config.data.output_csv == "test.csv"
        assert config.data.dataset_name == "test_dataset"
        assert isinstance(config.worker, WorkerConfig)
        assert isinstance(config.retrieval, RetrievalConfig)
        assert isinstance(config.ml, MLConfig)
        assert isinstance(config.feature, FeatureConfig)
        assert isinstance(config.database, DatabaseConfig)

    def test_alligator_config_with_column_types(self):
        """Test AlligatorConfig with column types."""
        df = pd.DataFrame({"col1": [1, 2]})
        column_types = {"0": ["Q5"], "1": "Q11424"}

        config = AlligatorConfig(input_csv=df, output_csv="test.csv", column_types=column_types)

        assert config.data.column_types["0"] == ["Q5"]
        assert config.data.column_types["1"] == ["Q11424"]

    def test_alligator_config_to_dict(self):
        """Test AlligatorConfig to_dict method."""
        df = pd.DataFrame({"col1": [1, 2]})
        config = AlligatorConfig(input_csv=df, output_csv="test.csv")
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "data" in config_dict
        assert "worker" in config_dict
        assert "retrieval" in config_dict
        assert "ml" in config_dict
        assert "feature" in config_dict
        assert "database" in config_dict

    def test_alligator_config_validation(self):
        """Test AlligatorConfig validation."""
        df = pd.DataFrame({"col1": [1, 2]})
        config = AlligatorConfig(input_csv=df, output_csv="test.csv")
        assert config.validate() is True

    def test_alligator_config_repr(self):
        """Test AlligatorConfig string representation."""
        df = pd.DataFrame({"col1": [1, 2]})
        config = AlligatorConfig(input_csv=df, output_csv="test.csv")
        repr_str = repr(config)
        assert "AlligatorConfig" in repr_str
        assert "data=" in repr_str
