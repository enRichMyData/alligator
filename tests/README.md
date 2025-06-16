# Alligator Test Suite

This directory contains comprehensive test cases for the Alligator entity linking system.

## Structure

```
tests/
├── conftest.py          # Common test fixtures and configuration
├── test_config.py       # Tests for configuration management
├── test_types.py        # Tests for data types and models
├── test_fetchers.py     # Tests for API fetchers (candidates, objects, literals)
├── test_processors.py   # Tests for row processing logic
├── test_alligator.py    # Tests for main Alligator class
├── test_utils.py        # Tests for utility functions
└── README.md           # This file
```

## Running Tests

### Quick Start

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=alligator --cov-report=html

# Run specific test file
python -m pytest tests/test_config.py

# Run specific test
python -m pytest tests/test_config.py::TestDataConfig::test_dataconfig_with_dataframe
```

### Using the Test Runner Script

```bash
# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py --coverage

# Run only unit tests
python run_tests.py --unit

# Run fast tests only (exclude slow tests)
python run_tests.py --fast

# Run with verbose output
python run_tests.py --verbose

# Run specific test file
python run_tests.py tests/test_config.py
```

### Test Categories

Tests are organized with markers:

- `@pytest.mark.unit` - Fast unit tests (default for most tests)
- `@pytest.mark.integration` - Integration tests that test component interaction
- `@pytest.mark.slow` - Slow-running tests

Run specific categories:

```bash
# Unit tests only
python -m pytest -m unit

# Integration tests only
python -m pytest -m integration

# Exclude slow tests
python -m pytest -m "not slow"
```

## Test Coverage

### What's Tested

✅ **Configuration Management (`test_config.py`)**
- DataConfig validation and processing
- WorkerConfig defaults and customization
- RetrievalConfig endpoint configuration
- MLConfig model path handling
- FeatureConfig validation
- DatabaseConfig connection settings
- AlligatorConfig integration and validation
- Column types processing and validation

✅ **Data Types (`test_types.py`)**
- Entity creation and validation
- RowData structure and handling
- Candidate serialization/deserialization
- Type conversions and edge cases

✅ **API Fetchers (`test_fetchers.py`)**
- CandidateFetcher with caching and retry logic
- ObjectFetcher for relationship data
- LiteralFetcher for literal values
- HTTP request mocking and error handling
- Cache key generation and management
- Async batch processing

✅ **Row Processing (`test_processors.py`)**
- Entity extraction from documents
- Candidate fetching with column types
- Batch processing orchestration
- Fuzzy retry logic
- Data deduplication

✅ **Main Alligator Class (`test_alligator.py`)**
- Initialization with various configurations
- Configuration delegation
- Backward compatibility
- Method delegation to coordinator
- Error handling for invalid inputs

✅ **Utilities (`test_utils.py`)**
- ColumnHelper index normalization and validation
- String cleaning and normalization
- Edge cases and error conditions

### Mocking Strategy

All tests use proper mocking to isolate components:

- **Database connections** - Mocked to avoid requiring MongoDB
- **HTTP requests** - Mocked to avoid external API dependencies
- **File system** - Uses temporary files for safe testing
- **Async operations** - Properly handled with pytest-asyncio

### Test Fixtures

Common fixtures in `conftest.py`:

- `sample_dataframe` - Sample pandas DataFrame
- `temp_csv_file` - Temporary CSV file for testing
- `sample_column_types` - Example column type configurations
- `mock_mongo_wrapper` - Mocked database wrapper
- `mock_aiohttp_session` - Mocked HTTP session

## Writing New Tests

### Guidelines

1. **Use descriptive test names** - `test_processor_handles_empty_cells_correctly`
2. **One assertion per concept** - Test one thing at a time
3. **Use fixtures for common setup** - Leverage existing fixtures in conftest.py
4. **Mock external dependencies** - Don't rely on databases or APIs
5. **Test both success and failure cases** - Include error conditions
6. **Use appropriate markers** - Mark slow tests as `@pytest.mark.slow`

### Example Test Structure

```python
import pytest
from unittest.mock import Mock, patch
from alligator.module import ClassToTest

class TestClassToTest:
    @pytest.fixture
    def instance(self, mock_dependency):
        """Create instance with mocked dependencies."""
        return ClassToTest(dependency=mock_dependency)

    def test_method_success_case(self, instance):
        """Test successful operation."""
        result = instance.method("input")
        assert result == "expected_output"

    def test_method_error_case(self, instance):
        """Test error handling."""
        with pytest.raises(ValueError):
            instance.method("invalid_input")

    @pytest.mark.asyncio
    async def test_async_method(self, instance):
        """Test async method."""
        result = await instance.async_method()
        assert result is not None
```

## Continuous Integration

These tests are designed to run in CI environments:

- No external dependencies (databases, APIs)
- Deterministic (no random behavior)
- Fast execution (unit tests < 1s each)
- Comprehensive coverage of core functionality

## Troubleshooting

### Common Issues

1. **ImportError** - Make sure the alligator package is installed: `pip install -e .`
2. **ModuleNotFoundError** - Run tests from the project root directory
3. **Async test failures** - Ensure pytest-asyncio is installed and configured
4. **MongoDB errors** - Tests should not connect to real MongoDB (check mocking)

### Debugging

```bash
# Run with output capture disabled to see print statements
python -m pytest tests/test_file.py -s

# Run with pdb on failures
python -m pytest tests/test_file.py --pdb

# Run single test with maximum verbosity
python -m pytest tests/test_file.py::test_name -vvv
```
