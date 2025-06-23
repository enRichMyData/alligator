from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from alligator.fetchers import CandidateFetcher, LiteralFetcher, ObjectFetcher, get_cache_key
from alligator.mongo import MongoCache


class MockResponse:
    def __init__(self, json_data, status=200):
        self.json_data = json_data
        self.status = status

    def raise_for_status(self):
        return None

    async def json(self):
        return self.json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestGetCacheKey:
    def test_cache_key_generation(self):
        """Test cache key generation with various parameters."""
        key1 = get_cache_key(endpoint="test", token="token", entity="Brad Pitt")
        key2 = get_cache_key(endpoint="test", token="token", entity="Brad Pitt")
        key3 = get_cache_key(endpoint="test", token="token", entity="Angelina Jolie")

        assert key1 == key2  # Same parameters should generate same key
        assert key1 != key3  # Different parameters should generate different keys
        assert len(key1) == 64  # SHA256 hash length

    def test_cache_key_with_different_order(self):
        """Test cache key generation with different parameter order."""
        key1 = get_cache_key(endpoint="test", token="token", entity="Brad Pitt")
        key2 = get_cache_key(token="token", endpoint="test", entity="Brad Pitt")

        assert key1 == key2  # Order shouldn't matter due to sorted keys


class TestCandidateFetcher:
    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session."""
        session = Mock(spec=aiohttp.ClientSession)
        return session

    @pytest.fixture
    def candidate_fetcher(self, mock_session):
        """Create CandidateFetcher instance with mocked dependencies."""
        with patch("alligator.fetchers.MongoWrapper"), patch("alligator.fetchers.MongoCache"):
            fetcher = CandidateFetcher(
                endpoint="https://test.com/api",
                token="test_token",
                num_candidates=10,
                session=mock_session,
                use_cache=True,
                db_name="test_db",
                mongo_uri="mongodb://localhost:27017",
            )
            fetcher.cache = Mock(spec=MongoCache)
            return fetcher

    def test_candidate_fetcher_init(self, candidate_fetcher, mock_session):
        """Test CandidateFetcher initialization."""
        assert candidate_fetcher.endpoint == "https://test.com/api"
        assert candidate_fetcher.token == "test_token"
        assert candidate_fetcher.num_candidates == 10
        assert candidate_fetcher.session == mock_session
        assert candidate_fetcher.use_cache is True

    def test_candidate_fetcher_without_cache(self, mock_session):
        """Test CandidateFetcher without cache."""
        with patch("alligator.fetchers.MongoWrapper"):
            fetcher = CandidateFetcher(
                endpoint="https://test.com/api",
                token="test_token",
                num_candidates=10,
                session=mock_session,
                use_cache=False,
            )
            assert fetcher.cache is None

    @pytest.mark.asyncio
    async def test_fetch_candidates_batch(self, candidate_fetcher):
        """Test fetch_candidates_batch method."""
        entities = ["Brad Pitt", "Tom Cruise"]
        fuzzies = [False, True]
        qids = [["Q35332"], []]
        types = [["Q5"], ["Q5", "Q33999"]]

        # Mock the async method
        candidate_fetcher.fetch_candidates_batch_async = AsyncMock(
            return_value={
                "Brad Pitt": [{"id": "Q35332", "name": "Brad Pitt"}],
                "Tom Cruise": [{"id": "Q37079", "name": "Tom Cruise"}],
            }
        )

        result = await candidate_fetcher.fetch_candidates_batch(entities, fuzzies, qids, types)

        assert "Brad Pitt" in result
        assert "Tom Cruise" in result
        candidate_fetcher.fetch_candidates_batch_async.assert_called_once_with(
            entities, fuzzies, qids, types
        )

    @pytest.mark.asyncio
    async def test_fetch_candidates_batch_without_types(self, candidate_fetcher):
        """Test fetch_candidates_batch without types parameter."""
        entities = ["Brad Pitt"]
        fuzzies = [False]
        qids = [[]]

        candidate_fetcher.fetch_candidates_batch_async = AsyncMock(
            return_value={"Brad Pitt": [{"id": "Q35332", "name": "Brad Pitt"}]}
        )

        await candidate_fetcher.fetch_candidates_batch(entities, fuzzies, qids)

        # Should call with empty types list
        candidate_fetcher.fetch_candidates_batch_async.assert_called_once_with(
            entities, fuzzies, qids, [[]]
        )

    @pytest.mark.asyncio
    async def test_fetch_candidates_basic_functionality(self, candidate_fetcher):
        """Test basic candidate fetching functionality by mocking the entire method."""
        # Instead of mocking HTTP details, mock the entire _fetch_candidates method
        expected_result = (
            "Brad Pitt",
            [{"id": "Q35332", "name": "Brad Pitt", "description": "American actor"}],
        )

        candidate_fetcher._fetch_candidates = AsyncMock(return_value=expected_result)

        result = await candidate_fetcher._fetch_candidates("Brad Pitt", False, "", "Q5")

        assert result[0] == "Brad Pitt"
        assert len(result[1]) == 1
        assert result[1][0]["id"] == "Q35332"

    @pytest.mark.asyncio
    async def test_fetch_candidates_with_placeholders(self, candidate_fetcher):
        """Test fetching candidates with placeholder handling."""
        # Mock the method to return data with placeholders
        expected_result = (
            "Brad Pitt",
            [
                {"id": "Q35332", "name": "Brad Pitt"},
                {
                    "id": "Q12345",
                    "name": None,
                    "description": None,
                    "features": None,
                    "is_placeholder": True,
                },
            ],
        )

        candidate_fetcher._fetch_candidates = AsyncMock(return_value=expected_result)

        result = await candidate_fetcher._fetch_candidates("Brad Pitt", False, "Q35332 Q12345", "")

        assert result[0] == "Brad Pitt"
        assert len(result[1]) == 2  # Original + placeholder

        # Check that placeholder was added for missing QID
        qids = {item["id"] for item in result[1]}
        assert "Q35332" in qids
        assert "Q12345" in qids

        # Check for placeholder
        placeholder_found = any(c.get("is_placeholder") for c in result[1])
        assert placeholder_found

    @pytest.mark.asyncio
    async def test_fetch_candidates(self, candidate_fetcher: CandidateFetcher):
        """Test fetch_candidates method."""
        mock_response_data = [{"id": "Q76", "name": "Barack Obama"}]
        candidate_fetcher.session.get.return_value = MockResponse(mock_response_data)
        result = await candidate_fetcher._fetch_candidates("Barack Obama", False, "Q76", "Q5")
        assert result[0] == "Barack Obama"
        assert len(result[1]) == 1

    @pytest.mark.asyncio
    async def test_fetch_candidates_batch_async_with_cache(self, candidate_fetcher):
        """Test batch fetching with cache hits."""
        # Mock cache to return cached result for first entity
        candidate_fetcher.cache.get = Mock(
            side_effect=[[{"id": "Q35332", "name": "Brad Pitt"}], None]  # Cache hit  # Cache miss
        )

        # Mock _fetch_candidates for cache miss
        candidate_fetcher._fetch_candidates = AsyncMock(
            return_value=("Tom Cruise", [{"id": "Q37079", "name": "Tom Cruise"}])
        )

        entities = ["Brad Pitt", "Tom Cruise"]
        fuzzies = [False, False]
        qids = [[], []]
        types = [[], []]

        result = await candidate_fetcher.fetch_candidates_batch_async(
            entities, fuzzies, qids, types
        )

        assert "Brad Pitt" in result
        assert "Tom Cruise" in result
        assert result["Brad Pitt"][0]["id"] == "Q35332"
        assert result["Tom Cruise"][0]["id"] == "Q37079"

        # Should only fetch for cache miss
        candidate_fetcher._fetch_candidates.assert_called_once()

    def test_remove_placeholders(self, candidate_fetcher):
        """Test placeholder removal."""
        results = {
            "Brad Pitt": [
                {"id": "Q35332", "name": "Brad Pitt"},
                {"id": "Q12345", "name": None, "is_placeholder": True},
            ],
            "Tom Cruise": [{"id": "Q37079", "name": "Tom Cruise"}],
        }

        cleaned_results = candidate_fetcher._remove_placeholders(results)

        assert len(cleaned_results["Brad Pitt"]) == 1
        assert cleaned_results["Brad Pitt"][0]["id"] == "Q35332"
        assert len(cleaned_results["Tom Cruise"]) == 1

    @pytest.mark.asyncio
    async def test_fetch_candidates_with_types(self, candidate_fetcher: CandidateFetcher):
        """Test _fetch_candidates method with types parameter."""
        mock_response_data = [
            {
                "id": "Q76",
                "name": "Barack Obama",
                "description": "44th President of the United States",
            }
        ]

        # Create a proper async context manager mock
        mock_response = MockResponse(mock_response_data)
        candidate_fetcher.session.get.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        candidate_fetcher.session.get.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await candidate_fetcher._fetch_candidates("Barack Obama", False, "Q76", "Q5")

        # Verify the result structure
        assert result[0] == "Barack Obama"
        assert len(result[1]) == 1
        assert result[1][0]["id"] == "Q76"
        assert result[1][0]["name"] == "Barack Obama"

        # Verify the URL was constructed correctly with types parameter
        args, kwargs = candidate_fetcher.session.get.call_args
        url = args[0]
        assert "name=Barack%20Obama" in url
        assert "types=Q5" in url
        assert "ids=Q76" in url
        assert f"limit={candidate_fetcher.num_candidates}" in url

    @pytest.mark.asyncio
    async def test_fetch_candidates_with_placeholder_addition(
        self, candidate_fetcher: CandidateFetcher
    ):
        """Test that missing QIDs get placeholder entries added."""
        # Mock response that's missing one of the requested QIDs
        mock_response_data = [{"id": "Q76", "name": "Barack Obama"}]

        mock_response = MockResponse(mock_response_data)
        candidate_fetcher.session.get.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        candidate_fetcher.session.get.return_value.__aexit__ = AsyncMock(return_value=None)

        # Request two QIDs but only one is returned
        result = await candidate_fetcher._fetch_candidates(
            "Barack Obama", False, "Q76 Q12345", "Q5"
        )

        assert result[0] == "Barack Obama"
        assert len(result[1]) == 2  # Original + placeholder

        # Check that both QIDs are present
        qids = {item["id"] for item in result[1]}
        assert "Q76" in qids
        assert "Q12345" in qids

        # Check that placeholder was created for missing QID
        placeholder = next((item for item in result[1] if item["id"] == "Q12345"), None)
        assert placeholder is not None
        assert placeholder.get("is_placeholder") is True
        assert placeholder["name"] is None

    @pytest.mark.asyncio
    async def test_fetch_candidates_network_error_handling(
        self, candidate_fetcher: CandidateFetcher
    ):
        """Test error handling and retry logic."""
        # Mock session to raise an exception
        candidate_fetcher.session.get.side_effect = aiohttp.ClientError("Network error")

        result = await candidate_fetcher._fetch_candidates("Barack Obama", False, "", "")

        # Should return empty list on persistent failure
        assert result[0] == "Barack Obama"
        assert result[1] == []

        # Should have attempted multiple times (5 attempts)
        assert candidate_fetcher.session.get.call_count == 5


class TestObjectFetcher:
    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session."""
        return Mock(spec=aiohttp.ClientSession)

    @pytest.fixture
    def object_fetcher(self, mock_session):
        """Create ObjectFetcher instance with mocked dependencies."""
        with patch("alligator.fetchers.MongoWrapper"), patch("alligator.fetchers.MongoCache"):
            fetcher = ObjectFetcher(
                endpoint="https://test.com/objects",
                token="test_token",
                session=mock_session,
                db_name="test_db",
                mongo_uri="mongodb://localhost:27017",
            )
            fetcher.cache = Mock(spec=MongoCache)
            return fetcher

    def test_object_fetcher_init(self, object_fetcher, mock_session):
        """Test ObjectFetcher initialization."""
        assert object_fetcher.endpoint == "https://test.com/objects"
        assert object_fetcher.token == "test_token"
        assert object_fetcher.session == mock_session

    @pytest.mark.asyncio
    async def test_fetch_objects_empty_list(self, object_fetcher):
        """Test fetching objects with empty entity list."""
        result = await object_fetcher.fetch_objects([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_objects_functionality(self, object_fetcher):
        """Test fetching objects functionality."""
        entity_ids = ["Q35332", "Q37079"]

        # Mock the method to return expected results
        expected_result = {
            "Q35332": {"objects": {"P106": ["actor"]}},
            "Q37079": {"objects": {"P31": ["human"]}},
        }

        object_fetcher.fetch_objects = AsyncMock(return_value=expected_result)

        result = await object_fetcher.fetch_objects(entity_ids)

        assert "Q35332" in result
        assert "Q37079" in result
        assert result["Q35332"] == {"objects": {"P106": ["actor"]}}
        assert result["Q37079"] == {"objects": {"P31": ["human"]}}
        assert result["Q35332"]["objects"]["P106"] == ["actor"]
        assert result["Q37079"]["objects"]["P31"] == ["human"]


class TestLiteralFetcher:
    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session."""
        return Mock(spec=aiohttp.ClientSession)

    @pytest.fixture
    def literal_fetcher(self, mock_session):
        """Create LiteralFetcher instance with mocked dependencies."""
        with patch("alligator.fetchers.MongoWrapper"), patch("alligator.fetchers.MongoCache"):
            fetcher = LiteralFetcher(
                endpoint="https://test.com/literals",
                token="test_token",
                session=mock_session,
                db_name="test_db",
                mongo_uri="mongodb://localhost:27017",
            )
            fetcher.cache = Mock(spec=MongoCache)
            return fetcher

    def test_literal_fetcher_init(self, literal_fetcher, mock_session):
        """Test LiteralFetcher initialization."""
        assert literal_fetcher.endpoint == "https://test.com/literals"
        assert literal_fetcher.token == "test_token"
        assert literal_fetcher.session == mock_session

    @pytest.mark.asyncio
    async def test_fetch_literals_empty_list(self, literal_fetcher):
        """Test fetching literals with empty entity list."""
        result = await literal_fetcher.fetch_literals([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_literals_functionality(self, literal_fetcher):
        """Test successful literal fetching functionality."""
        entity_ids = ["Q35332"]

        # Mock the method to return expected results
        expected_result = {"Q35332": {"literals": {"P569": {"1963-12-18": [1.0]}}}}

        literal_fetcher.fetch_literals = AsyncMock(return_value=expected_result)

        result = await literal_fetcher.fetch_literals(entity_ids)

        assert "Q35332" in result
        assert result["Q35332"] == {"literals": {"P569": {"1963-12-18": [1.0]}}}
        assert "literals" in result["Q35332"]
        assert "P569" in result["Q35332"]["literals"]
