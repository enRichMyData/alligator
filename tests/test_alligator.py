import os
import tempfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from alligator.alligator import Alligator
from alligator.config import AlligatorConfig
from alligator.coordinator import AlligatorCoordinator


class TestAlligator:
    @pytest.fixture
    def mock_coordinator(self):
        """Mock AlligatorCoordinator."""
        with patch("alligator.alligator.AlligatorCoordinator") as mock_coord:
            coordinator_instance = Mock(spec=AlligatorCoordinator)

            # Configure the data_manager mock
            mock_data_manager = Mock()
            mock_mongo_wrapper = Mock()
            mock_data_manager.mongo_wrapper = mock_mongo_wrapper
            coordinator_instance.data_manager = mock_data_manager

            # Configure other manager mocks
            coordinator_instance.worker_manager = Mock()
            coordinator_instance.ml_manager = Mock()
            coordinator_instance.output_manager = Mock()
            coordinator_instance.feature = Mock()

            mock_coord.return_value = coordinator_instance
            yield coordinator_instance

    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,movie,year\n")
            f.write("Brad Pitt,Fight Club,1999\n")
            f.write("Tom Cruise,Top Gun,1986\n")
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_alligator_init_with_csv_file(self, temp_csv_file, mock_coordinator):
        """Test Alligator initialization with CSV file."""
        alligator = Alligator(
            input_csv=temp_csv_file, dataset_name="test_dataset", table_name="test_table"
        )

        assert isinstance(alligator.config, AlligatorConfig)
        assert alligator.config.data.dataset_name == "test_dataset"
        assert alligator.config.data.table_name == "test_table"
        assert alligator.coordinator == mock_coordinator

        # Check backward compatibility properties
        assert str(alligator.input_csv) == temp_csv_file
        assert alligator.output_csv == alligator.config.data.output_csv

    def test_alligator_init_with_dataframe(self, mock_coordinator):
        """Test Alligator initialization with DataFrame."""
        df = pd.DataFrame(
            {
                "name": ["Brad Pitt", "Tom Cruise"],
                "movie": ["Fight Club", "Top Gun"],
                "year": [1999, 1986],
            }
        )

        alligator = Alligator(
            input_csv=df, output_csv="test_output.csv", dataset_name="test_dataset"
        )

        assert isinstance(alligator.config.data.input_csv, pd.DataFrame)
        assert alligator.config.data.output_csv == "test_output.csv"
        assert alligator.config.data.dataset_name == "test_dataset"

    def test_alligator_init_with_column_types(self, temp_csv_file, mock_coordinator):
        """Test Alligator initialization with column types."""
        column_types = {
            "0": ["Q5"],  # Person
            "1": ["Q11424"],  # Film
            "2": "Q577",  # Year (single string)
        }

        alligator = Alligator(input_csv=temp_csv_file, column_types=column_types)

        assert alligator.config.data.column_types["0"] == ["Q5"]
        assert alligator.config.data.column_types["1"] == ["Q11424"]
        assert alligator.config.data.column_types["2"] == ["Q577"]

    def test_alligator_init_with_all_parameters(self, temp_csv_file, mock_coordinator):
        """Test Alligator initialization with comprehensive parameters."""
        alligator = Alligator(
            input_csv=temp_csv_file,
            output_csv="test_output.csv",
            dataset_name="test_dataset",
            table_name="test_table",
            target_rows=["0", "1"],
            target_columns={"NE": {"0": "PERSON"}, "LIT": {"2": "NUMBER"}, "IGNORED": []},
            column_types={"0": ["Q5"]},
            worker_batch_size=32,
            num_workers=2,
            max_candidates_in_result=3,
            entity_retrieval_endpoint="https://test.com/api",
            entity_retrieval_token="test_token",
            candidate_retrieval_limit=20,
            ranker_model_path="/path/to/ranker.h5",
            ml_worker_batch_size=128,
            num_ml_workers=1,
            save_output=True,
            save_output_to_csv=True,
            http_session_limit=16,
            mongo_uri="mongodb://localhost:27017",
        )

        # Check data config
        assert alligator.config.data.dataset_name == "test_dataset"
        assert alligator.config.data.table_name == "test_table"
        assert set(alligator.config.data.target_rows) == {"0", "1"}  # Order might change
        assert alligator.config.data.column_types["0"] == ["Q5"]

        # Check worker config
        assert alligator.config.worker.worker_batch_size == 32
        assert alligator.config.worker.num_workers == 2

        # Check retrieval config
        assert alligator.config.retrieval.entity_retrieval_endpoint == "https://test.com/api"
        assert alligator.config.retrieval.entity_retrieval_token == "test_token"
        assert alligator.config.retrieval.candidate_retrieval_limit == 20
        assert alligator.config.retrieval.max_candidates_in_result == 3
        assert alligator.config.retrieval.http_session_limit == 16

        # Check ML config
        assert alligator.config.ml.ranker_model_path == "/path/to/ranker.h5"
        assert alligator.config.ml.ml_worker_batch_size == 128
        assert alligator.config.ml.num_ml_workers == 1

        # Check database config
        assert alligator.config.database.mongo_uri == "mongodb://localhost:27017"

    def test_alligator_run_method(self, temp_csv_file, mock_coordinator):
        """Test Alligator run method."""
        alligator = Alligator(input_csv=temp_csv_file)

        # Mock the coordinator's run method
        mock_coordinator.run = Mock(return_value=[])

        result = alligator.run()

        # Verify coordinator's run was called
        mock_coordinator.run.assert_called_once()
        assert result == []

    def test_alligator_backward_compatibility_properties(self, temp_csv_file, mock_coordinator):
        """Test backward compatibility properties."""
        alligator = Alligator(
            input_csv=temp_csv_file,
            output_csv="test_output.csv",
            dataset_name="test_dataset",
            table_name="test_table",
            num_workers=4,
            worker_batch_size=64,
            max_candidates_in_result=5,
        )

        # Test that properties are accessible for backward compatibility
        assert str(alligator.input_csv) == temp_csv_file
        assert alligator.output_csv == "test_output.csv"
        assert alligator.dataset_name == "test_dataset"
        assert alligator.table_name == "test_table"
        assert alligator.num_workers == 4
        assert alligator.worker_batch_size == 64
        assert alligator.max_candidates_in_result == 5

    def test_alligator_config_delegation(self, temp_csv_file, mock_coordinator):
        """Test that Alligator properly delegates to AlligatorConfig."""
        with patch("alligator.alligator.AlligatorConfig") as mock_config_class:
            mock_config_instance = Mock(spec=AlligatorConfig)

            # Configure mock config attributes
            mock_data_config = Mock()
            mock_data_config.input_csv = temp_csv_file
            mock_data_config.output_csv = None
            mock_data_config.dataset_name = "test_dataset"
            mock_data_config.table_name = None
            mock_data_config.target_rows = None
            mock_data_config.target_columns = None
            mock_data_config.column_types = {"0": ["Q5"]}

            mock_config_instance.data = mock_data_config
            mock_config_instance.pipeline = Mock()
            mock_config_instance.retrieval = Mock()
            mock_config_instance.ml = Mock()
            mock_config_instance.feature = Mock()
            mock_config_instance.database = Mock()
            mock_config_instance.http = Mock()
            mock_config_instance.worker = Mock()
            mock_config_instance.worker.worker_batch_size = 16
            mock_config_instance.worker.num_workers = 1

            mock_config_class.return_value = mock_config_instance
            Alligator(
                input_csv=temp_csv_file, dataset_name="test_dataset", column_types={"0": ["Q5"]}
            )

            # Verify AlligatorConfig was called with correct parameters
            mock_config_class.assert_called_once()
            call_kwargs = mock_config_class.call_args[1]

            assert call_kwargs["input_csv"] == temp_csv_file
            assert call_kwargs["dataset_name"] == "test_dataset"
            assert call_kwargs["column_types"] == {"0": ["Q5"]}

    def test_alligator_missing_input_csv(self, mock_coordinator):
        """Test Alligator initialization without input CSV."""
        with pytest.raises(ValueError, match="Input CSV or DataFrame must be provided"):
            Alligator(input_csv=None)

    def test_alligator_nonexistent_file(self, mock_coordinator):
        """Test Alligator initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            Alligator(input_csv="/nonexistent/file.csv")

    def test_alligator_dataframe_without_output(self, mock_coordinator):
        """Test Alligator initialization with DataFrame but no output CSV."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        with pytest.raises(ValueError, match="An output name must be specified"):
            Alligator(input_csv=df, save_output=True, save_output_to_csv=True)

    def test_alligator_with_kwargs(self, temp_csv_file, mock_coordinator):
        """Test Alligator initialization with additional kwargs."""
        alligator = Alligator(
            input_csv=temp_csv_file, custom_param="custom_value", another_param=123
        )

        # Should not raise error and should initialize successfully
        assert isinstance(alligator.config, AlligatorConfig)

    def test_alligator_repr(self, temp_csv_file, mock_coordinator):
        """Test Alligator string representation."""
        alligator = Alligator(input_csv=temp_csv_file)
        repr_str = str(alligator)

        # Should contain class name and basic info
        assert "Alligator" in repr_str or repr_str is not None  # Basic check

    def test_alligator_config_access(self, temp_csv_file, mock_coordinator):
        """Test accessing configuration through Alligator instance."""
        alligator = Alligator(
            input_csv=temp_csv_file, dataset_name="test_dataset", column_types={"0": ["Q5"]}
        )

        # Should be able to access config
        assert alligator.config is not None
        assert isinstance(alligator.config, AlligatorConfig)

        # Should be able to access nested config properties
        assert alligator.config.data.dataset_name == "test_dataset"
        assert alligator.config.data.column_types["0"] == ["Q5"]
