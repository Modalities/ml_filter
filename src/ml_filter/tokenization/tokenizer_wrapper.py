from abc import ABC
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizer


class TokenizerWrapper(ABC):
    """Abstract interface for tokenizers."""

    # TODO: check return type
    def apply_tokenizer_chat_template(self, prompt: List[Dict[str, str]], tokenize: bool) -> str | List[int]:
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
        add_generation_prompt: bool,
        truncation: Optional[bool] = False,
        padding: Optional[bool | str] = False,
        max_length: Optional[int] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initializes the PreTrainedHFTokenizer.

        Args:
            pretrained_model_name_or_path (str): Name or path of the pretrained model.
            add_generation_prompt (bool):  If this is set, a prompt with the token(s) that indicate
                the start of an assistant message will be appended to the formatted output.
                This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
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
        self.add_generation_prompt = add_generation_prompt

    @property
    def pad_token(self) -> str:
        return self.tokenizer.pad_token

    @property
    def eos_token(self) -> str:
        return self.tokenizer.eos_token

    def apply_tokenizer_chat_template(self, prompt: List[Dict[str, str]], tokenize: bool) -> str | List[int]:
        """Applies a chat template to the given prompt.

        Args:
            prompt (List[Dict[str, str]]): The prompt
            tokenize (bool): Wether to tokenize the prompt or just apply the chat template

        Returns:
            str | List[int]: The chat-tempalte-applied prompt as list of int if tokenize is True otherwise str.
        """
        return self.tokenizer.apply_chat_template(
            prompt, tokenize=tokenize, add_generation_prompt=self.add_generation_prompt
        )

    def __getattr__(self, name: str) -> Any:
        """Delegates missing method calls to the underlying Hugging Face tokenizer."""
        return getattr(self.tokenizer, name)
