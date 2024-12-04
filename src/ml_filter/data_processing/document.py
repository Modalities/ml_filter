from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel


class DocumentProcessingStatus(str, Enum):
    """An enumeration representing the status of the document processing operation."""

    SUCCESS = "success"
    ERROR_SERVER = "error_server"
    ERROR_NO_GENERATED_TEXT = "error_no_generated_text"
    ERROR_FAULTY_SCORE = "error_faulty_score"


class DocumentProcessingTags(str, Enum):
    """An enumeration representing the tags for the document processing operation."""

    TRUNCATED = "truncated"
    DETOKENIZATION_MISMATCH = "detokenization_mismatch"


@dataclass
class ProcessedDocument:
    """A class representing a model response for a given document."""

    document_id: str
    raw_data_file_path: Path
    original_text: str
    original_history: List[Dict[str, str]] = field(default_factory=list)
    preprocessed_text: str = ""
    prompt: str = ""
    generated_text: str = ""
    score_type: str = ""
    score: float = None
    original_score: float = None
    document_processing_status: DocumentProcessingStatus = None
    errors: List[str] = field(default_factory=list)
    tags: List[DocumentProcessingTags] = field(default_factory=list)
    document_text_detokenized: str = ""
    truncated_preprocessed_text: str = ""
    timestamp: int = 0


class MetaInformation(BaseModel):
    """A class representing the meta information for a given document."""

    prompt: str
    prompt_lang: str
    model: str
    raw_data_file_path: str


class Annotation(BaseModel):
    """A class representing the output document from the model."""

    document_id: str
    scores: List[float | None] = []
    explanations: List[str] = []
    errors: List[List[str]] = []
    time_stamps: List[int] = []
    document_processing_status: List[DocumentProcessingStatus] = []
    meta_information: MetaInformation
