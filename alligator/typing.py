from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict

LitType = Literal["NUMBER", "STRING", "DATETIME"]
NerType = Literal["LOCATION", "ORGANIZATION", "PERSON", "OTHER"]


class ColType(TypedDict):
    NE: Dict[str, LitType | NerType]
    LIT: Dict[str, LitType | NerType]
    IGNORED: List[str]


@dataclass
class Entity:
    """Represents a named entity from a table cell."""

    value: str
    row_index: Optional[int]
    col_index: str  # Stored as string for consistency
    context_text: str
    correct_qid: Optional[str] = None
    fuzzy: bool = False


@dataclass
class RowData:
    """Represents a row's data with all necessary context."""

    doc_id: str
    row: List[Any]
    ne_columns: Dict[str, str]
    lit_columns: Dict[str, str]
    context_columns: List[str]
    correct_qids: Dict[str, str]
    row_index: Optional[int]
    context_text: str
    row_hash: str
