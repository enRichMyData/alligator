from functools import lru_cache
from typing import List, Set

from nltk.tokenize import word_tokenize

from alligator import STOP_WORDS


class ColumnHelper:
    """Helper class for consistent column index handling."""

    @staticmethod
    def normalize(col_idx) -> str:
        """Convert any column index to its canonical string form."""
        return str(col_idx)

    @staticmethod
    def to_int(col_idx) -> int:
        """Convert a column index to integer for list access."""
        return int(str(col_idx))

    @staticmethod
    def is_valid_index(col_idx, row_length: int) -> bool:
        """Check if a column index is valid for a given row."""
        try:
            idx = ColumnHelper.to_int(col_idx)
            return 0 <= idx < row_length
        except (ValueError, TypeError):
            return False


@lru_cache(maxsize=10000)
def ngrams(string: str, n: int = 3) -> List[str]:
    tokens: List[str] = [string[i : i + n] for i in range(len(string) - n + 1)]
    return tokens


@lru_cache(maxsize=10000)
def tokenize_text(text: str) -> Set[str]:
    tokens: List[str] = word_tokenize(text.lower())
    return {t for t in tokens if t not in STOP_WORDS}
