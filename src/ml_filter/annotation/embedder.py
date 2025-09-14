# load hf key and set cache dir
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

load_dotenv()


def bytes_to_gib_str(bytes_val):
    return f"{bytes_val / 1024 ** 3:.2f} GiB"


class GteMultilingualBase:
    """
    A wrapper class for the 'Alibaba-NLP/gte-multilingual-base' embedding model.

    Attributes:
        device (torch.device or str): The device on which to load the model (e.g., 'cuda', 'cpu').
        dtype (torch.dtype): The data type for model computations (default: torch.bfloat16).
        tokenizer (AutoTokenizer): The tokenizer associated with the GTE model.
        model (AutoModel): The loaded GTE-Multilingual-Base model.
    """

    def __init__(self, device, dtype=torch.bfloat16):
        """
        Initializes the GteMultilingualBase model.

        Args:
            device (torch.device or str): The device to load the model onto (e.g., 'cuda', 'cpu').
            dtype (torch.dtype, optional): The data type for model operations. Defaults to torch.bfloat16.
        """

        self.device = device
        self.dtype = dtype

        model_id = "Alibaba-NLP/gte-multilingual-base"

        # Load the tokenizer specific to the embedding model.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Load the pre-trained GTE-Multilingual-Base model.
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,  # Allows loading custom code from the model's repository.
            torch_dtype=dtype,  # Sets the data type for model parameters and computations.
            unpad_inputs=True,  # Optimizes for unpadded inputs if applicable.
            use_memory_efficient_attention=True,  # Leverages memory-efficient attention mechanisms.
        ).to(
            device
        )  # Move the model to the specified device.

    def embed(self, texts):
        """
        Generates embeddings for a list of text strings.

        Args:
            texts (list[str]): A list of text strings to embed.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, embedding_dim)
                          containing the normalized embeddings for the input texts.
        """

        # Tokenize the input texts, ensuring proper padding, truncation, and tensor conversion.
        batch_tokens = self.tokenizer(
            texts,
            max_length=8192,  # Maximum sequence length for tokenization.
            padding="longest",  # Pad to the length of the longest sequence in the batch.
            truncation=True,  # Truncate sequences longer than max_length.
            return_tensors="pt",  # Return PyTorch tensors.
        ).to(
            self.device
        )  # Move tokens to the specified device.

        with torch.no_grad():  # Disable gradient calculation for inference to save memory and speed up computation.
            output = self.model(**batch_tokens)

        # Extract the embeddings from the CLS token (first token) of the last hidden state.
        embeddings = output.last_hidden_state[:, 0]
        # Normalize the embeddings to unit length (L2 normalization).
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class SnowflakeArcticEmbedMV2_0:
    """
    A wrapper class for the 'Snowflake/snowflake-arctic-embed-m-v2.0' embedding model.


    Attributes:
        device (torch.device or str): The device on which to load the model.
        dtype (torch.dtype): The data type for model computations (default: torch.bfloat16).
        tokenizer (AutoTokenizer): The tokenizer for the Snowflake Arctic Embed model.
        model (AutoModel): The loaded Snowflake Arctic Embed M v2.0 model.
    """

    def __init__(self, device, dtype=torch.bfloat16, compile=False):
        """
        Initializes the SnowflakeArcticEmbedMV2_0 model.

        Args:
            device (torch.device or str): The device to load the model onto.
            dtype (torch.dtype, optional): The data type for model operations. Defaults to torch.bfloat16.
            compile (bool, optional): Whether to compile the model's forward pass using `torch.compile`
                                      for potential performance improvements. Defaults to False.
        """

        self.device = device
        self.dtype = dtype

        model_id = "Snowflake/snowflake-arctic-embed-m-v2.0"

        # print device
        print(f"Using device: {device} for Snowflake Arctic Embed M v2.0 model")

        # Load the tokenizer specific to the embedding model.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Load the pre-trained Snowflake Arctic Embed M v2.0 model.
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,  # Allows loading custom code from the model's repository.
            torch_dtype=dtype,  # Sets the data type for model parameters and computations.
            unpad_inputs=True,  # Optimizes for unpadded inputs if applicable.
            # device_map={'': device},     # Maps the model to the specified device.
            add_pooling_layer=False,  # Prevents adding an extra pooling layer if not needed.
            use_memory_efficient_attention=True,  # Leverages memory-efficient attention mechanisms.
        )
        self.model.to(device)  # Move the model to the specified device.
        self.model.eval()  # Set the model to evaluation mode.

        # Compile the model's forward pass if `compile` is True.
        if compile:
            self.model.forward = torch.compile(self.model.forward)

    def embed(self, texts):
        """
        Generates embeddings for a list of text strings using the Snowflake Arctic Embed model.

        Args:
            texts (list[str]): A list of text strings to embed.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, embedding_dim)
                          containing the normalized embeddings.
        """

        batch_tokens = self.tokenizer(
            texts,
            max_length=8192,  # Maximum sequence length for tokenization.
            padding="longest",  # Pad to the length of the longest sequence in the batch.
            truncation=True,  # Truncate sequences longer than max_length.
            return_tensors="pt",
        )  # Return PyTorch tensors.

        # batch_tokens = texts
        batch_tokens = {
            k: v.to(torch.device(self.device)) for k, v in batch_tokens.items()
        }  # Move tokens to the specified device.

        with torch.no_grad(), torch.cuda.device(self.device):
            output = self.model(**batch_tokens)

            # Extract and normalize the embeddings.
            # embeddings = output.last_hidden_state[:, 0]
            embeddings = F.normalize(output.last_hidden_state[:, 0], p=2, dim=1)

        embeddings = embeddings.cpu().tolist()
        # torch.cuda.empty_cache()  # Clear CUDA memory cache to free up resources.

        return embeddings


class JinaEmbeddingsV3TextMatching:
    """
    A wrapper class for the 'jinaai/jina-embeddings-v3' model, specifically
    configured for 'text-matching' tasks.

    Attributes:
        device (torch.device or str): The device on which to load the model.
        dtype (torch.dtype): The data type for model computations (default: torch.bfloat16).
        model (AutoModel): The loaded Jina Embeddings V3 model.
    """

    def __init__(self, device, dtype=torch.bfloat16):
        """
        Initializes the JinaEmbeddingsV3TextMatching model.

        Args:
            device (torch.device or str): The device to load the model onto.
            dtype (torch.dtype, optional): The data type for model operations. Defaults to torch.bfloat16.
        """

        self.device = device
        self.dtype = dtype

        model_id = "jinaai/jina-embeddings-v3"

        # Load the Jina Embeddings V3 model.
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,  # Allows loading custom code from the model's repository.
            torch_dtype=torch.bfloat16,  # Specifically set dtype to bfloat16 for this model.
        ).to(
            device
        )  # Move the model to the specified device.

    def embed(self, texts):
        """
        Generates embeddings for a list of text strings using the Jina Embeddings V3 model
        with a 'text-matching' task.
        Args:
            texts (list[str]): A list of text strings to embed.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, embedding_dim)
                          containing the embeddings.
        """

        with torch.no_grad():  # Disable gradient calculation for inference.
            # Use the model's built-in encode method with 'text-matching' task.
            output = self.model.encode(texts, task="text-matching")

        # Convert the output (which might be a numpy array or list) to a PyTorch tensor
        # and move it to the specified device and data type.
        embeddings = torch.tensor(output).to(self.device, self.dtype)

        return embeddings


class Qwen3Embedder:
    """
    A wrapper class for the 'Qwen/Qwen3-Embedding-8B' embedding model.

    This class implements the Qwen 3 embedding model with last token pooling strategy
    and left padding tokenization, following the established patterns of other embedding
    models in the codebase.

    Attributes:
        device (torch.device or str): The device on which to load the model.
        dtype (torch.dtype): The data type for model computations (default: torch.bfloat16).
        tokenizer (AutoTokenizer): The tokenizer configured with left padding.
        model (AutoModel): The loaded Qwen3-Embedding-8B model.
    """

    def __init__(self, device, dtype=torch.bfloat16):
        """
        Initializes the Qwen_3_Embedder model.

        Args:
            device (torch.device or str): The device to load the model onto (e.g., 'cuda', 'cpu').
            dtype (torch.dtype, optional): The data type for model operations. Defaults to torch.bfloat16.
        """

        self.device = device
        self.dtype = dtype

        model_id = "Qwen/Qwen3-Embedding-0.6B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"  # Set left padding as recommended by Qwen model

        # Load the pre-trained Qwen3-Embedding-8B model
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).to(device)

        self.model.eval()  # Set the model to evaluation mode

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extracts embeddings from the last token position for each sequence in the batch.

        This method handles both left and right padding scenarios:
        - For left padding: Returns the last token embeddings (rightmost position)
        - For right padding: Returns embeddings at sequence-specific end positions based on attention mask

        Args:
            last_hidden_states (torch.Tensor): Hidden states from the model's last layer.
                Shape: (batch_size, sequence_length, hidden_size)
            attention_mask (torch.Tensor): Attention mask indicating valid tokens.
                Shape: (batch_size, sequence_length)

        Returns:
            torch.Tensor: Pooled embeddings from the last token of each sequence.
                Shape: (batch_size, hidden_size)
        """

        # Check if all sequences end at the same position (indicating left padding)
        # This is done by checking if the sum of attention mask equals sequence_length for all sequences
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]

        if left_padding:
            # For left padding, return the last token (rightmost position) for all sequences
            return last_hidden_states[:, -1]
        else:
            # For right padding, find the last valid token position for each sequence
            # attention_mask.sum(dim=1) gives the length of each sequence
            # Subtract 1 to get the index of the last valid token (0-indexed)
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]

            # Select the last valid token for each sequence
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """
        Formats task instructions for the Qwen embedding model.

        This method creates a standardized instruction format that combines
        a task description with a specific query, following the Qwen model's
        expected input format for optimal embedding generation.

        Args:
            task_description (str): Description of the task or context for the embedding.
            query (str): The specific text query to be embedded.

        Returns:
            str: Formatted instruction string in the format:
                 "Instruct: {task_description}\nQuery: {query}"

        Example:
            >>> embedder.get_detailed_instruct("Retrieve semantically similar text", "machine learning")
            'Instruct: Retrieve semantically similar text\\nQuery: machine learning'
        """
        return f"Instruct: {task_description}\nQuery:{query}"

    def embed(self, texts):
        """
        Generates embeddings for a list of text strings using the Qwen 3 embedding model.

        This method tokenizes the input texts with left padding, processes them through
        the model, extracts embeddings using last token pooling, and applies L2 normalization.

        Args:
            texts (list[str]): A list of text strings to embed.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, embedding_dim)
                          containing the normalized embeddings for the input texts.
        """

        # Tokenize the input texts with Qwen-specific configuration
        batch_tokens = self.tokenizer(
            texts,
            max_length=32768,  # Maximum sequence length as specified for Qwen model
            padding=True,  # Enable padding to handle variable length sequences
            truncation=True,  # Truncate sequences longer than max_length
            return_tensors="pt",  # Return PyTorch tensors
        ).to(
            self.device
        )  # Move tokens to the specified device

        with torch.no_grad():  # Disable gradient calculation for inference
            output = self.model(**batch_tokens)
            # Extract embeddings using last token pooling strategy
            embeddings = self.last_token_pool(output.last_hidden_state, batch_tokens["attention_mask"])

            # Apply L2 normalization to the embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()


def get_embedder_instance(model_id, device, dtype):
    """
    Factory function to get an instance of a specified embedding model.

    This function dynamically creates an instance of the appropriate embedding
    model class based on the provided `model_id`. This allows for flexible
    selection and instantiation of different embedding models, which is crucial
    for evaluating and comparing various multilingual embedding approaches
    as might be done in the research presented in the associated paper.

    Args:
        model_id (str): The identifier of the embedding model to instantiate.
                        Supported IDs include:
                        - 'Alibaba-NLP/gte-multilingual-base'
                        - 'Snowflake/snowflake-arctic-embed-m-v2.0'
                        - 'jinaai/jina-embeddings-v3'
                        - 'embedding_at_scale' (for Qwen/Qwen3-Embedding-8B)
        device (torch.device or str): The device to load the model onto.
        dtype (torch.dtype): The data type for model operations.

    Returns:
        Union[GteMultilingualBase, SnowflakeArcticEmbedMV2_0, JinaEmbeddingsV3TextMatching, Qwen_3_Embedder]:
            An instance of the requested embedding model class.

    Raises:
        ValueError: If an unknown `model_id` is provided.
    """

    if model_id == "Alibaba-NLP/gte-multilingual-base":
        embedder_class = GteMultilingualBase

    elif model_id == "Snowflake/snowflake-arctic-embed-m-v2.0":
        embedder_class = SnowflakeArcticEmbedMV2_0

    elif model_id == "jinaai/jina-embeddings-v3":
        embedder_class = JinaEmbeddingsV3TextMatching

    elif model_id == "Qwen/Qwen3-Embedding-0.6B":
        embedder_class = Qwen3Embedder

    else:
        raise ValueError(f"Unknown model ID: {model_id}")

    return embedder_class(device, dtype)
