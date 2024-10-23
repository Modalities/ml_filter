from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


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
    original_text: str
    original_history: List[Dict[str, str]] = field(default_factory=list)
    preprocessed_text: str = ""
    prompt: str = ""
    generated_text: str = ""
    score_type: str = ""
    score: float = None
    document_processing_status: DocumentProcessingStatus = None
    errors: List[str] = field(default_factory=list)
    tags: List[DocumentProcessingTags] = field(default_factory=list)