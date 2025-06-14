import pandas as pd
import pytest

from alligator.utils import ColumnHelper, clean_str


class TestColumnHelper:
    """Test ColumnHelper utility class."""

    def test_normalize_string_index(self):
        """Test normalizing string column indices."""
        assert ColumnHelper.normalize("0") == "0"
        assert ColumnHelper.normalize("1") == "1"
        assert ColumnHelper.normalize("10") == "10"

    def test_normalize_integer_index(self):
        """Test normalizing integer column indices."""
        assert ColumnHelper.normalize(0) == "0"
        assert ColumnHelper.normalize(1) == "1"
        assert ColumnHelper.normalize(10) == "10"

    def test_normalize_negative_index(self):
        """Test normalizing negative indices."""
        assert ColumnHelper.normalize(-1) == "-1"
        assert ColumnHelper.normalize("-1") == "-1"

    def test_to_int_valid_string(self):
        """Test converting valid string indices to int."""
        assert ColumnHelper.to_int("0") == 0
        assert ColumnHelper.to_int("1") == 1
        assert ColumnHelper.to_int("10") == 10

    def test_to_int_negative_string(self):
        """Test converting negative string indices to int."""
        assert ColumnHelper.to_int("-1") == -1
        assert ColumnHelper.to_int("-10") == -10

    def test_to_int_invalid_string(self):
        """Test converting invalid string indices to int raises ValueError."""
        with pytest.raises(ValueError):
            ColumnHelper.to_int("abc")
        with pytest.raises(ValueError):
            ColumnHelper.to_int("")
        with pytest.raises(ValueError):
            ColumnHelper.to_int("1.5")

    def test_is_valid_index_positive(self):
        """Test validating positive indices."""
        assert ColumnHelper.is_valid_index("0", 5) is True
        assert ColumnHelper.is_valid_index("4", 5) is True
        assert ColumnHelper.is_valid_index("5", 5) is False

    def test_is_valid_index_negative(self):
        """Test validating negative indices."""
        # Current implementation doesn't support negative indices
        assert ColumnHelper.is_valid_index("-1", 5) is False

    def test_is_valid_index_edge_cases(self):
        """Test edge cases for index validation."""
        # Empty list
        assert ColumnHelper.is_valid_index("0", 0) is False
        assert ColumnHelper.is_valid_index("-1", 0) is False

        # Single element list
        assert ColumnHelper.is_valid_index("0", 1) is True
        assert ColumnHelper.is_valid_index("1", 1) is False
        # Current implementation doesn't support negative indices
        assert ColumnHelper.is_valid_index("-1", 1) is False

    def test_is_valid_index_invalid_strings(self):
        """Test validating invalid string indices."""
        assert ColumnHelper.is_valid_index("abc", 5) is False
        assert ColumnHelper.is_valid_index("", 5) is False
        assert ColumnHelper.is_valid_index("1.5", 5) is False

    def test_column_helper_integration(self):
        """Test ColumnHelper methods work together."""
        # Simulate processing a column index
        original_index = 2
        normalized = ColumnHelper.normalize(original_index)
        assert normalized == "2"

        is_valid = ColumnHelper.is_valid_index(normalized, 5)
        assert is_valid is True

        converted_back = ColumnHelper.to_int(normalized)
        assert converted_back == original_index

    def test_column_helper_with_pandas_index(self):
        """Test ColumnHelper with pandas-like scenarios."""
        # Simulate working with a DataFrame with 3 columns
        num_columns = 3

        # Valid column indices (only positive indices are supported)
        valid_indices = ["0", "1", "2"]
        for idx in valid_indices:
            assert ColumnHelper.is_valid_index(idx, num_columns) is True

        # Invalid column indices
        invalid_indices = ["3", "4", "-1", "-2", "-3"]
        for idx in invalid_indices:
            assert ColumnHelper.is_valid_index(idx, num_columns) is False

    def test_column_helper_type_conversions(self):
        """Test type conversion edge cases."""
        # Test with None (should raise error)
        with pytest.raises((ValueError, TypeError)):
            ColumnHelper.to_int(None)

        # Test normalize with None
        assert ColumnHelper.normalize(None) == "None"


class TestCleanStr:
    """Test clean_str utility function."""

    def test_clean_str_basic(self):
        """Test basic string cleaning."""
        assert clean_str("Hello World") == "hello world"
        assert clean_str("UPPERCASE") == "uppercase"

    def test_clean_str_with_special_characters(self):
        """Test cleaning strings with special characters."""
        assert clean_str("hello, world!") == "hello, world!"
        # clean_str replaces underscores with spaces
        assert clean_str("hello-world_test") == "hello-world test"

    def test_clean_str_empty_and_none(self):
        """Test cleaning empty strings and None values."""
        assert clean_str("") == ""
        # clean_str doesn't strip whitespace-only strings to empty
        assert clean_str("   ") == "   "

    def test_clean_str_with_brackets(self):
        """Test cleaning strings with numerical content in brackets."""
        # Should remove numerical content in square brackets (not parentheses)
        assert clean_str("test [123] string") == "test string"
        assert (
            clean_str("item (456) description") == "item (456) description"
        )  # Parentheses not removed

    def test_clean_str_with_underscores(self):
        """Test cleaning strings with underscores."""
        assert clean_str("hello_world") == "hello world"
        assert clean_str("test_string_here") == "test string here"

    def test_clean_str_multiple_spaces(self):
        """Test cleaning strings with multiple spaces."""
        assert clean_str("hello    world") == "hello world"
        assert clean_str("  spaced  out  ") == "spaced out"

    def test_clean_str_edge_cases(self):
        """Test edge cases for clean_str."""
        # Test with numbers
        assert clean_str("123") == "123"
        assert clean_str("test123") == "test123"

        # Test with special characters that should remain
        assert clean_str("hello-world") == "hello-world"
        assert clean_str("test.com") == "test.com"

    def test_clean_str_pandas_na_values(self):
        """Test clean_str with pandas NA-like values."""
        # clean_str converts pandas.NA to string representation
        assert clean_str(pd.NA) == "<na>"

    def test_clean_str_none_values(self):
        """Test clean_str with None values."""
        assert clean_str(None) == "none"

    def test_clean_str_numeric_values(self):
        """Test clean_str with numeric input."""
        assert clean_str(123) == "123"
        assert clean_str(123.45) == "123.45"
        assert clean_str(0) == "0"

    def test_clean_str_boolean_values(self):
        """Test clean_str with boolean input."""
        assert clean_str(True) == "true"
        assert clean_str(False) == "false"

    def test_clean_str_fallback_behavior(self):
        """Test clean_str fallback to original when result is empty."""
        # Test case where cleaning results in empty string
        original = "[123]"  # This should be cleaned to empty, then fall back to original
        result = clean_str(original)
        # The function should return the original lowercase if cleaning results in empty
        assert result == "[123]"  # Falls back to original lowercase
