from typing import List, Optional

from pydantic import BaseModel, DirectoryPath, Field, FilePath


class OpenAIConfig(BaseModel):
    api_key: str = Field(..., description="OpenAI API key, typically sourced from an environment variable.")
    base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL.")

    class Config:
        extra = "forbid"


class PathsConfig(BaseModel):
    raw_data_file_paths: List[FilePath] = Field(default_factory=list)
    output_directory_path: DirectoryPath
    prompt_template_file_path: FilePath
    start_indexes: List[int] = Field(default_factory=list)

    class Config:
        extra = "forbid"


class SettingsConfig(BaseModel):
    model_name: str = Field(..., description="Model name (e.g., 'google/gemma-2-9b-it' for local, 'gpt-4o' for OpenAI).")
    num_gpus: int = Field(..., ge=0, description="Number of GPUs for local LLM (ignored for OpenAI).")
    tokenizer_name_or_path: str = Field(..., description="Tokenizer name or path.")
    paths: PathsConfig
    provider: str = Field(..., description="LLM provider: 'local' or 'openai'.")
    openai: Optional[OpenAIConfig] = Field(
        default=None, description="OpenAI-specific configuration, required if provider is 'openai'."
    )

    class Config:
        extra = "forbid"


class LLMRestClientConfig(BaseModel):
    max_tokens: int = Field(..., ge=1, description="Maximum total tokens (input + output).")
    max_pool_connections: int = Field(..., ge=1, description="Maximum pool connections for local LLM.")
    max_pool_maxsize: int = Field(..., ge=1, description="Maximum pool size for local LLM.")
    max_retries: int = Field(..., ge=0, description="Maximum number of retries for API requests.")
    backoff_factor: float = Field(..., ge=0.0, description="Backoff factor for retry delays.")
    timeout: int = Field(..., ge=1, description="Request timeout in seconds.")
    verbose: bool = Field(..., description="Enable verbose logging.")
    num_gpus: int = Field(..., ge=0, description="Number of GPUs for local LLM (ignored for OpenAI).")
    sampling_params: dict = Field(..., description="Sampling parameters for text generation.")


class TokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str = Field(..., description="Pretrained tokenizer name or path.")
    special_tokens: Optional[dict] = Field(default=None, description="Special tokens for tokenizer.")
    add_generation_prompt: bool = Field(..., description="Whether to add generation prompt.")

    class Config:
        extra = "forbid"


class PromptBuilderConfig(BaseModel):
    prompt_template_file_path: str = Field(..., description="Path to the prompt template file.")
    max_prompt_length: int = Field(..., ge=1, description="Maximum length of the prompt.")

    class Config:
        extra = "forbid"


class DocumentProcessorConfig(BaseModel):
    output_directory_path: DirectoryPath = Field(..., description="Output directory for processed documents.")
    queue_size: int = Field(..., ge=1, description="Size of the processing queue.")
    num_processes: int = Field(..., ge=1, description="Number of processes for document processing.")
    score_metric_name: str = Field(..., description="Name of the score metric.")
    strings_to_remove: List[str] = Field(default_factory=list, description="Strings to remove from documents.")
    jq_language_pattern: str = Field(..., description="JQ pattern for language metadata.")

    class Config:
        extra = "forbid"


class AnnotationPipelineConfig(BaseModel):
    settings: SettingsConfig = Field(..., description="General settings for the pipeline.")
    llm_rest_client: LLMRestClientConfig = Field(..., description="Configuration for LLM REST client.")
    tokenizer: TokenizerConfig = Field(..., description="Configuration for tokenizer.")
    prompt_builder: PromptBuilderConfig = Field(..., description="Configuration for prompt builder.")
    document_processor: DocumentProcessorConfig = Field(..., description="Configuration for document processor.")

    class Config:
        extra = "forbid"