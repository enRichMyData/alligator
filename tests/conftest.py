"""
Pytest configuration and common fixtures for alligator tests.
"""

import asyncio
import os
import tempfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "name": ["Brad Pitt", "Tom Cruise", "Angelina Jolie"],
            "movie": ["Fight Club", "Top Gun", "Maleficent"],
            "year": [1999, 1986, 2014],
            "genre": ["Drama", "Action", "Fantasy"],
        }
    )


@pytest.fixture
def temp_csv_file(sample_dataframe):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_dataframe.to_csv(f, index=False)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_column_types():
    """Sample column types for testing."""
    return {
        "0": ["Q5"],  # Person
        "1": ["Q11424"],  # Film
        "2": "Q577",  # Year (single string)
        "3": ["Q201658"],  # Genre
    }


@pytest.fixture
def sample_target_columns():
    """Sample target columns configuration."""
    return {
        "NE": {"0": "PERSON", "1": "OTHER"},
        "LIT": {"2": "NUMBER", "3": "STRING"},
        "IGNORED": [],
    }


@pytest.fixture
def sample_docs():
    """Sample document data for testing processors."""
    return [
        {
            "_id": "doc1",
            "row": ["Brad Pitt", "Fight Club", "1999", "Drama"],
            "row_index": 0,
            "ne_columns": {"0": "PERSON", "1": "OTHER"},
            "lit_columns": {"2": "NUMBER", "3": "STRING"},
            "context_columns": ["0", "1", "2", "3"],
            "correct_qids": {"0-0": ["Q35332"]},
        },
        {
            "_id": "doc2",
            "row": ["Tom Cruise", "Top Gun", "1986", "Action"],
            "row_index": 1,
            "ne_columns": {"0": "PERSON", "1": "OTHER"},
            "lit_columns": {"2": "NUMBER", "3": "STRING"},
            "context_columns": ["0", "1", "2", "3"],
            "correct_qids": {"1-0": ["Q37079"]},
        },
    ]


@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    from alligator.types import Entity

    return [
        Entity(value="Brad Pitt", row_index=0, col_index="0", correct_qids=["Q35332"]),
        Entity(value="Tom Cruise", row_index=1, col_index="0", correct_qids=["Q37079"]),
        Entity(value="Fight Club", row_index=0, col_index="1", correct_qids=[]),
    ]


@pytest.fixture
def sample_candidates():
    """Sample candidates for testing."""
    from alligator.types import Candidate

    return [
        Candidate(
            id="Q35332",
            name="Brad Pitt",
            score=0.95,
            description="American actor",
            kind="entity",
            NERtype="PERSON",
            features={"similarity": 0.95, "popularity": 0.8},
        ),
        Candidate(
            id="Q37079",
            name="Tom Cruise",
            score=0.92,
            description="American actor",
            kind="entity",
            NERtype="PERSON",
            features={"similarity": 0.92, "popularity": 0.9},
        ),
    ]


@pytest.fixture
def mock_mongo_wrapper():
    """Mock MongoWrapper for testing."""
    with patch("alligator.mongo.MongoWrapper") as mock_wrapper:
        instance = Mock()
        mock_wrapper.return_value = instance
        instance.get_db.return_value = Mock()
        instance.create_indexes.return_value = None
        instance.log_to_db.return_value = None
        yield instance


@pytest.fixture
def mock_mongo_cache():
    """Mock MongoCache for testing."""
    with patch("alligator.mongo.MongoCache") as mock_cache:
        instance = Mock()
        mock_cache.return_value = instance
        instance.get.return_value = None
        instance.put.return_value = None
        yield instance


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp ClientSession for testing."""
    session = Mock()
    session.get = Mock()
    session.post = Mock()
    session.close = Mock()
    return session


@pytest.fixture
def mock_feature():
    """Mock Feature object for testing."""
    from alligator.feature import Feature

    feature = Mock(spec=Feature)
    feature.selected_features = ["similarity", "popularity", "description_similarity"]
    return feature


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable logging during tests to reduce noise."""
    import logging

    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location/name."""
    for item in items:
        # Mark all tests in test_* files as unit tests by default
        if "test_" in item.nodeid and not any(
            marker.name in ["integration", "slow"] for marker in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)
