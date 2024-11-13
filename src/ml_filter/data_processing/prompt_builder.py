import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ml_filter.data_processing.document import DocumentProcessingTags, ProcessedDocument
from ml_filter.tokenizer.tokenizer_wrapper import TokenizerWrapper
from ml_filter.utils.string_comparison import get_char_differences


class PromptBuilder:
    """A class representing a prompt builder."""

    def __init__(self, prompt_path: Path, max_prompt_length: int = None, tokenizer: TokenizerWrapper = None) -> None:
        if max_prompt_length is not None and tokenizer is None:
            raise ValueError("If max_prompt_length is provided, tokenizer must also be provided.")

        self.max_prompt_length = max_prompt_length
        self.tokenizer = tokenizer
        with open(prompt_path, "r") as file:
            self.prompt_template = yaml.safe_load(file)["prompt"]

    def construct_prompt_helper(
        self, text: str, history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Constructs a prompt.

        Args:
            text (str): The text to be used as a placeholder in the prompt.
            history (Optional[List[Dict[str, str]]]): The history of prompts. Defaults to None.

        Returns:
            List[Dict[str, str]]: The prompt that is represtend as a list (history) of messages.
        """

        # TODO: Is this fixed for all models?
        prompt = {"role": "user", "content": self.prompt_template.format(placeholder=text)}

        if history is None:
            history = []

        history_new = copy.deepcopy(history)
        history_new.append(prompt)
        return history_new

    def construct_prompt(self, processed_document: ProcessedDocument) -> ProcessedDocument:
        if self.max_prompt_length is not None:
            prompt_empty = self.construct_prompt_helper("", processed_document.original_history)
            chat_template_length = len(self.tokenizer.apply_tokenizer_chat_template(prompt_empty, tokenize=True))

            document_text_tokenized = self.tokenizer.tokenizer.encode(processed_document.preprocessed_text)
            # remove tokens that exceed max_prompt_length
            if len(document_text_tokenized) > self.max_prompt_length - chat_template_length:
                document_text_tokenized = document_text_tokenized[: self.max_prompt_length - chat_template_length]
                # detokenize the tokens
                # we need to skip special tokens because they are not part of the original text e.g., begin of document token
                document_text_detokenized = self.tokenizer.tokenizer.decode(
                    document_text_tokenized, skip_special_tokens=True
                )
                # construct the prompt
                prompt_dict = self.construct_prompt_helper(
                    document_text_detokenized, processed_document.original_history
                )
                processed_document.tags.append(DocumentProcessingTags.TRUNCATED)
                truncated_preprocessed_text = processed_document.preprocessed_text[: len(document_text_detokenized)]
                if truncated_preprocessed_text != document_text_detokenized:
                    num_diff_chars = get_char_differences(truncated_preprocessed_text, document_text_detokenized)
                    logging.warning(
                        f"document {processed_document.document_id}: The truncated and detokenized text does not match the original text. Number of different characters: {num_diff_chars}"
                    )
                    processed_document.tags.append(DocumentProcessingTags.DETOKENIZATION_MISMATCH)
                    processed_document.errors.append(
                        f"Detokenization mismatch: Number of different characters: {num_diff_chars}"
                    )
                    processed_document.truncated_preprocessed_text = truncated_preprocessed_text
                    processed_document.document_text_detokenized = document_text_detokenized
                prompt_string = self.tokenizer.apply_tokenizer_chat_template(prompt_dict, tokenize=False)
                processed_document.prompt = prompt_string
                return processed_document

        prompt_dict = self.construct_prompt_helper(
            processed_document.preprocessed_text, processed_document.original_history
        )
        prompt_string = self.tokenizer.apply_tokenizer_chat_template(prompt_dict, tokenize=False)
        processed_document.prompt = prompt_string
        return processed_document
