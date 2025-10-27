from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, DirectoryPath, Field, FilePath, field_validator


class PathsConfig(BaseModel):
    raw_data_file_paths: List[FilePath] = Field(default_factory=list)
    output_directory_path: DirectoryPath
    prompt_template_file_path: FilePath
    start_indexes: List[int] = Field(default_factory=list)

    @field_validator("output_directory_path", mode="before")
    @classmethod
    def create_directory_if_not_exists(cls, v: str | Path) -> str | Path:
        """
        Validates the output directory path. If the directory does not exist,
        it is created.
        """
        # The input 'v' is the raw value (e.g., a string) before any other validation.
        if isinstance(v, str):
            Path(v).mkdir(parents=True, exist_ok=True)
        elif isinstance(v, Path):
            v.mkdir(parents=True, exist_ok=True)

        # Return the original value to be processed by Pydantic's DirectoryPath validator.
        return v


class SettingsConfig(BaseModel):
    model_name: str
    num_gpus: int
    tokenizer_name_or_path: str
    paths: PathsConfig


class LLMRestClientConfig(BaseModel):
    model_name: str
    max_tokens: int
    max_pool_connections: int
    max_pool_maxsize: int
    max_retries: int
    backoff_factor: float
    timeout: int
    verbose: bool
    num_gpus: int
    sampling_params: dict


class TokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str
    special_tokens: Optional[dict]
    add_generation_prompt: bool


class PromptBuilderConfig(BaseModel):
    prompt_template_file_path: str
    max_prompt_length: int


class DocumentProcessorConfig(BaseModel):
    output_directory_path: DirectoryPath
    queue_size: int
    num_processes: int
    score_metric_name: str
    strings_to_remove: List[str] = Field(default_factory=list)
    jq_language_pattern: str
    document_id_column: str


class AnnotationPipelineConfig(BaseModel):
    settings: SettingsConfig
    llm_rest_client: LLMRestClientConfig
    tokenizer: TokenizerConfig
    prompt_builder: PromptBuilderConfig
    document_processor: DocumentProcessorConfig
