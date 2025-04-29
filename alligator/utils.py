import re
from functools import lru_cache
from typing import List, Set

from dateutil.parser import parse
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


def clean_str(value):
    original_value = str(value)

    # Remove content within brackets (including the brackets themselves)
    value = re.sub(r"\[.*?\]", "", original_value)

    # Remove content within parentheses (including the parentheses themselves)
    value = re.sub(r"\(.*?\)", "", value)

    # Replace specific unwanted characters with space
    stop_characters = ["_"]
    for char in stop_characters:
        value = value.replace(char, " ")

    # Remove extra spaces and strip leading/trailing spaces
    value = " ".join(value.split())

    # Return the original string if the cleaned result is empty
    if not value.strip():
        return original_value

    return value


def parse_date(str_date):
    date_parsed = None

    try:
        int(str_date)
        str_date = f"{str_date}-01-01"
    except Exception:
        pass

    try:
        date_parsed = parse(str_date)
    except Exception:
        pass

    if date_parsed is not None:
        return date_parsed

    try:
        str_date = str_date[1:]
        date_parsed = parse(str_date)
    except Exception:
        pass

    if date_parsed is not None:
        return date_parsed

    try:
        year = str_date.split("-")[0]
        str_date = f"{year}-01-01"
        date_parsed = parse(str_date)
    except Exception:
        pass

    return date_parsed


def perc_diff(v1, v2):
    diff = 1 - (abs(v1 - v2) / max(abs(v1), abs(v2), 1))
    diff = round(diff, 4)
    return diff


def word2ngrams(text, n=None):
    """Convert word into character ngrams."""
    if n is None:
        n = len(text)
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def get_ngrams(text, n=3):
    ngrams = set()
    for token in text.split(" "):
        temp = word2ngrams(token, n)
        for ngram in temp:
            ngrams.add(ngram)
    return set(ngrams)
