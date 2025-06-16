from unittest.mock import AsyncMock, Mock, patch

import pytest

from alligator.feature import Feature
from alligator.fetchers import CandidateFetcher
from alligator.processors import RowBatchProcessor
from alligator.types import Candidate, Entity, RowData


class TestRowBatchProcessor:
    @pytest.fixture
    def mock_feature(self):
        """Mock Feature object."""
        feature = Mock(spec=Feature)
        feature.selected_features = ["similarity", "popularity"]
        return feature

    @pytest.fixture
    def mock_candidate_fetcher(self):
        """Mock CandidateFetcher."""
        return Mock(spec=CandidateFetcher)

    @pytest.fixture
    def mock_mongo_wrapper(self):
        """Mock MongoWrapper."""
        with patch("alligator.processors.MongoWrapper") as mock_wrapper:
            yield mock_wrapper.return_value

    @pytest.fixture
    def processor(self, mock_feature, mock_candidate_fetcher, mock_mongo_wrapper):
        """Create RowBatchProcessor instance with mocked dependencies."""
        return RowBatchProcessor(
            dataset_name="test_dataset",
            table_name="test_table",
            feature=mock_feature,
            candidate_fetcher=mock_candidate_fetcher,
            max_candidates_in_result=5,
            column_types={"0": ["Q5"], "1": ["Q11424"]},
            db_name="test_db",
            mongo_uri="mongodb://localhost:27017",
        )

    def test_processor_initialization(self, processor, mock_feature, mock_candidate_fetcher):
        """Test RowBatchProcessor initialization."""
        assert processor.dataset_name == "test_dataset"
        assert processor.table_name == "test_table"
        assert processor.feature == mock_feature
        assert processor.candidate_fetcher == mock_candidate_fetcher
        assert processor.max_candidates_in_result == 5
        assert processor.column_types["0"] == ["Q5"]
        assert processor.column_types["1"] == ["Q11424"]

    def test_processor_column_types_processing(
        self, mock_feature, mock_candidate_fetcher, mock_mongo_wrapper
    ):
        """Test column types processing in processor initialization."""
        # Test with mixed string and list types
        processor = RowBatchProcessor(
            dataset_name="test",
            table_name="test",
            feature=mock_feature,
            candidate_fetcher=mock_candidate_fetcher,
            column_types={"0": "Q5", "1": ["Q11424", "Q515"]},  # type: ignore
        )

        assert processor.column_types["0"] == ["Q5"]
        assert processor.column_types["1"] == ["Q11424", "Q515"]

    def test_processor_without_column_types(
        self, mock_feature, mock_candidate_fetcher, mock_mongo_wrapper
    ):
        """Test processor initialization without column types."""
        processor = RowBatchProcessor(
            dataset_name="test",
            table_name="test",
            feature=mock_feature,
            candidate_fetcher=mock_candidate_fetcher,
        )

        assert len(processor.column_types) == 0

    def test_extract_entities_and_row_data(self, processor):
        """Test extracting entities and row data from documents."""
        docs = [
            {
                "_id": "doc1",
                "data": ["Brad Pitt", "Fight Club", "1999"],
                "row_id": 0,
                "classified_columns": {
                    "NE": {"0": "PERSON", "1": "OTHER"},
                    "LIT": {"2": "NUMBER"},
                },
                "context_columns": ["0", "1", "2"],
                "correct_qids": {"0-0": ["Q35332"]},
            },
            {
                "_id": "doc2",
                "data": ["Tom Cruise", "Top Gun", "1986"],
                "row_id": 1,
                "classified_columns": {
                    "NE": {"0": "PERSON", "1": "OTHER"},
                    "LIT": {"2": "NUMBER"},
                },
                "context_columns": ["0", "1", "2"],
                "correct_qids": {},
            },
        ]

        entities, row_data_list = processor._extract_entities(docs)

        # Check entities
        assert len(entities) == 4  # 2 entities per document
        assert all(isinstance(e, Entity) for e in entities)

        # Check first entity (values are cleaned/lowercased)
        assert entities[0].value == "brad pitt"
        assert entities[0].row_index == 0
        assert entities[0].col_index == "0"
        assert entities[0].correct_qids == ["Q35332"]

        # Check row data
        assert len(row_data_list) == 2
        assert all(isinstance(rd, RowData) for rd in row_data_list)
        assert row_data_list[0].doc_id == "doc1"
        assert row_data_list[0].row == ["brad pitt", "fight club", "1999"]  # Cleaned values

    def test_extract_entities_with_invalid_columns(self, processor):
        """Test extracting entities with invalid column indices."""
        docs = [
            {
                "_id": "doc1",
                "data": ["Brad Pitt", "Fight Club"],
                "row_id": 0,
                "classified_columns": {"NE": {"5": "PERSON"}, "LIT": {}},  # Invalid column index
                "context_columns": ["0", "1"],
                "correct_qids": {},
            }
        ]

        entities, row_data_list = processor._extract_entities(docs)

        # Should skip invalid column
        assert len(entities) == 0
        assert len(row_data_list) == 1

    def test_extract_entities_with_empty_cells(self, processor):
        """Test extracting entities with empty or NaN cell values."""
        docs = [
            {
                "_id": "doc1",
                "data": ["", "Fight Club", None],
                "row_id": 0,
                "classified_columns": {
                    "NE": {"0": "PERSON", "1": "OTHER", "2": "OTHER"},
                    "LIT": {},
                },
                "context_columns": ["0", "1", "2"],
                "correct_qids": {},
            }
        ]

        entities, row_data_list = processor._extract_entities(docs)

        # Should extract non-empty entities (empty string filtered out, None becomes "none")
        assert len(entities) == 2
        assert entities[0].value == "fight club"
        assert entities[1].value == "none"
        assert entities[0].col_index == "1"

    @pytest.mark.asyncio
    async def test_fetch_all_candidates(self, processor):
        """Test fetching candidates for all entities."""
        entities = [
            Entity(value="Brad Pitt", row_index=0, col_index="0", correct_qids=["Q35332"]),
            Entity(value="Tom Cruise", row_index=1, col_index="0", correct_qids=[]),
            Entity(value="Fight Club", row_index=0, col_index="1", correct_qids=[]),
        ]

        # Mock candidate fetcher
        processor.candidate_fetcher.fetch_candidates_batch = AsyncMock(
            return_value={
                "Brad Pitt": [{"id": "Q35332", "name": "Brad Pitt"}],
                "Tom Cruise": [{"id": "Q37079", "name": "Tom Cruise"}],
                "Fight Club": [{"id": "Q190050", "name": "Fight Club"}],
            }
        )

        # Mock logger
        processor.logger = Mock()
        processor.logger.info = Mock()

        candidates = await processor._fetch_all_candidates(entities)

        # Verify fetch was called with correct parameters
        call_args = processor.candidate_fetcher.fetch_candidates_batch.call_args
        entities_arg, fuzzies_arg, qids_arg, types_arg = call_args[1].values()

        assert len(entities_arg) == 3
        assert "Brad Pitt" in entities_arg
        assert "Tom Cruise" in entities_arg
        assert "Fight Club" in entities_arg

        # Check that column types are included
        brad_pitt_idx = entities_arg.index("Brad Pitt")
        fight_club_idx = entities_arg.index("Fight Club")

        assert types_arg[brad_pitt_idx] == ["Q5"]  # Column 0 type
        assert types_arg[fight_club_idx] == ["Q11424"]  # Column 1 type

        # Check candidates
        assert "Brad Pitt" in candidates
        assert len(candidates["Brad Pitt"]) == 1
        assert isinstance(candidates["Brad Pitt"][0], Candidate)

    @pytest.mark.asyncio
    async def test_fetch_all_candidates_with_fuzzy_retry(self, processor):
        """Test fetching candidates with fuzzy retry."""
        processor.fuzzy_retry = True

        entities = [Entity(value="Unknown Person", row_index=0, col_index="0", correct_qids=[])]

        # Mock initial fetch returns no candidates, retry returns some
        processor.candidate_fetcher.fetch_candidates_batch = AsyncMock(
            side_effect=[
                {"Unknown Person": []},  # Initial fetch - no results
                {"Unknown Person": [{"id": "Q123", "name": "Unknown Person"}]},  # Retry fetch
            ]
        )

        processor.logger = Mock()
        processor.logger.info = Mock()

        await processor._fetch_all_candidates(entities)

        # Should have made two calls - initial + retry
        assert processor.candidate_fetcher.fetch_candidates_batch.call_count == 2

        # Check retry call was made with fuzzy=True
        retry_call_args = processor.candidate_fetcher.fetch_candidates_batch.call_args_list[1]
        fuzzies_arg = retry_call_args[1]["fuzzies"]
        assert fuzzies_arg[0] is True  # Fuzzy should be True for retry

    @pytest.mark.asyncio
    async def test_fetch_all_candidates_deduplication(self, processor):
        """Test that duplicate mentions are deduplicated during fetching."""
        entities = [
            Entity(value="Brad Pitt", row_index=0, col_index="0"),
            Entity(value="Brad Pitt", row_index=1, col_index="0"),  # Duplicate
            Entity(value="Tom Cruise", row_index=2, col_index="0"),
        ]

        processor.candidate_fetcher.fetch_candidates_batch = AsyncMock(
            return_value={
                "Brad Pitt": [{"id": "Q35332", "name": "Brad Pitt"}],
                "Tom Cruise": [{"id": "Q37079", "name": "Tom Cruise"}],
            }
        )

        processor.logger = Mock()
        processor.logger.info = Mock()

        await processor._fetch_all_candidates(entities)

        # Should have called with only 2 unique entities
        call_args = processor.candidate_fetcher.fetch_candidates_batch.call_args
        entities_arg = call_args[1]["entities"]
        assert len(entities_arg) == 2
        assert "Brad Pitt" in entities_arg
        assert "Tom Cruise" in entities_arg

    @pytest.mark.asyncio
    async def test_process_rows_batch(self, processor):
        """Test processing a batch of rows."""
        docs = [
            {
                "_id": "doc1",
                "data": ["Brad Pitt", "Fight Club"],
                "row_id": 0,
                "classified_columns": {"NE": {"0": "PERSON", "1": "OTHER"}, "LIT": {}},
                "context_columns": ["0", "1"],
                "correct_qids": {},
            }
        ]

        # Mock the individual methods
        entities = [Entity(value="brad pitt", row_index=0, col_index="0")]  # cleaned value
        row_data_list = [
            RowData(
                doc_id="doc1",
                row=["brad pitt", "fight club"],  # cleaned values
                ne_columns={"0": "PERSON", "1": "OTHER"},
                lit_columns={},
                context_columns=["0", "1"],
                correct_qids={},
                row_index=0,
            )
        ]

        processor._extract_entities = Mock(return_value=(entities, row_data_list))
        processor._fetch_all_candidates = AsyncMock(
            return_value={
                "brad pitt": [Candidate(id="Q35332", name="Brad Pitt")]
            }  # key is cleaned
        )
        processor._process_rows = AsyncMock()

        await processor.process_rows_batch(docs)

        # Verify all methods were called
        processor._extract_entities.assert_called_once_with(docs)
        processor._fetch_all_candidates.assert_called_once_with(entities)
        processor._process_rows.assert_called_once()

    def test_processor_with_object_and_literal_fetchers(
        self, mock_feature, mock_candidate_fetcher, mock_mongo_wrapper
    ):
        """Test processor initialization with object and literal fetchers."""
        mock_object_fetcher = Mock()
        mock_literal_fetcher = Mock()

        processor = RowBatchProcessor(
            dataset_name="test",
            table_name="test",
            feature=mock_feature,
            candidate_fetcher=mock_candidate_fetcher,
            object_fetcher=mock_object_fetcher,
            literal_fetcher=mock_literal_fetcher,
            fuzzy_retry=True,
        )

        assert processor.object_fetcher == mock_object_fetcher
        assert processor.literal_fetcher == mock_literal_fetcher
        assert processor.fuzzy_retry is True
