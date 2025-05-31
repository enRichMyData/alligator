import re
from functools import lru_cache
from typing import Any, List, Set

import nltk
from dateutil.parser import parse
from nltk.tokenize import word_tokenize

from alligator import STOP_WORDS

RE_NUM_BRACKETS = re.compile(r"\[\d+\w*\]")


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


def keys_with_max_count(counter: dict) -> List[Any]:
    if not counter:
        return []
    max_val = max(counter.values())
    return [k for k, v in counter.items() if v == max_val]


@lru_cache(maxsize=10000)
def ngrams(string: str, n: int = 3) -> List[str]:
    tokens: List[str] = [string[i : i + n] for i in range(len(string) - n + 1)]
    return tokens


@lru_cache(maxsize=10000)
def tokenize_text(text: str) -> Set[str]:
    tokens: List[str] = word_tokenize(text.lower())
    return {t for t in tokens if t not in STOP_WORDS}


@lru_cache(maxsize=10000)
def clean_str(value):
    original_value = str(value).lower()
    value = original_value

    # Remove purely numerical content within brackets
    value = RE_NUM_BRACKETS.sub("", value)

    # Remove specific unwanted characters
    stop_characters = ["_"]
    for char in stop_characters:
        value = value.replace(char, " ")

    # Remove extra spaces and strip leading/trailing spaces
    value = " ".join(value.split())

    # Return the original string if the cleaned result is empty
    if not value:
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


@lru_cache(maxsize=10000)
def compute_similarity_between_string(str1: str, str2: str, ngram: int | None = None) -> float:
    ngrams_str1 = get_ngrams(str1, ngram)
    ngrams_str2 = get_ngrams(str2, ngram)
    score = len(ngrams_str1.intersection(ngrams_str2)) / max(len(ngrams_str1), len(ngrams_str2), 1)
    return score


@lru_cache(maxsize=10000)
def compute_similarity_between_string_token_based(str1: str, str2: str) -> float:
    token_set_str1 = set(str1.split(" "))
    token_set_str2 = set(str2.split(" "))
    score = len(token_set_str1.intersection(token_set_str2)) / max(
        len(token_set_str1), len(token_set_str2), 1
    )
    return score


@lru_cache(maxsize=10000)
def edit_distance(s1, s2):
    """
    Normalized Levhenstein distance function between two strings
    """
    return nltk.edit_distance(s1, s2) / max(len(s1), len(s2), 1)


def _my_abs(value1, value2):
    diff = 1 - (abs(value1 - value2) / max(abs(value1), abs(value2), 1))
    return diff


@lru_cache(maxsize=10000)
def compute_similarty_between_numbers(value1: str, value2: str) -> float:
    try:
        value1 = float(value1)
        value2 = float(value2)
        score = _my_abs(value1, value2)
    except Exception:
        score = 0

    return score


@lru_cache(maxsize=10000)
def compute_similarity_between_dates(date1: str, date2: str) -> float:
    try:
        date_parsed1 = parse_date(date1)
        date_parsed2 = parse_date(date2)
        score = (
            _my_abs(date_parsed1.year, date_parsed2.year)
            + _my_abs(date_parsed1.month, date_parsed2.month)
            + _my_abs(date_parsed1.day, date_parsed2.day)
        ) / 3
    except Exception:
        score = 0

    return score
