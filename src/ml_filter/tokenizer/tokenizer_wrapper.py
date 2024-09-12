from abc import ABC
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer


class TokenizerWrapper(ABC):
    """Abstract interface for tokenizers."""

    max_length: int
    truncation: bool

    # TODO: check return type
    def apply_tokenizer_chat_template(self, prompt: str, tokenize: bool) -> str:
        """Applies a chat template to the given prompt.

        Args:
            prompt (str): The prompt to apply the chat template to.
            tokenize (bool): Whether to tokenize the prompt.

        Returns:
            str: The prompt with the chat template applied.
        """
        raise NotImplementedError()

class PreTrainedHFTokenizer(TokenizerWrapper):
    """Wrapper for pretrained Hugging Face tokenizers."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        truncation: Optional[bool] = False,
        padding: Optional[bool | str] = False,
        max_length: Optional[int] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initializes the PreTrainedHFTokenizer.

        Args:
            pretrained_model_name_or_path (str): Name or path of the pretrained model.
            truncation (bool, optional): Flag whether to apply truncation. Defaults to False.
            padding (bool | str, optional): Defines the padding strategy. Defaults to False.
            max_length (int, optional): Maximum length of the tokenization output. Defaults to None.
            special_tokens (Dict[str, str], optional): Added token keys should be in the list
                of predefined special attributes: [bos_token, eos_token, unk_token, sep_token, pad_token,
                cls_token, mask_token, additional_special_tokens].
                Example: {"pad_token": "[PAD]"}
                Tokens are only added if they are not already in the vocabulary (tested by checking
                if the tokenizer assign the index of the unk_token to them). Defaults to None.
        """
        # also see here for the truncation and padding options and their effects:
        # https://huggingface.co/docs/transformers/pad_truncation#padding-and-truncation

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        if special_tokens is not None:
            # TODO check if we always want to set
            # replace_additional_special_tokens=False
            self.tokenizer.add_special_tokens(
                special_tokens_dict=special_tokens,
                replace_additional_special_tokens=False,
            )
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
    
    def apply_tokenizer_chat_template(self, prompt: List[Dict[str,str]], tokenize: bool) -> str:
        # TODO: check return type
        return self.tokenizer.apply_chat_template(prompt, tokenize=tokenize)
    
