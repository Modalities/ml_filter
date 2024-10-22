from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ml_filter.tokenizer.tokenizer_wrapper import TokenizerWrapper


class PromptBuilder:
    """A class representing a prompt builder."""

    def __init__(self, prompt_path: Path, max_prompt_length: int=None, tokenizer: TokenizerWrapper = None) -> None:
        if max_prompt_length is not None and tokenizer is None:
            raise ValueError("If max_prompt_length is provided, tokenizer must also be provided.")
        
        self.max_prompt_length = max_prompt_length
        self.tokenizer = tokenizer
        with open(prompt_path, "r") as file:
            self.prompt_template = yaml.safe_load(file)["prompt"]

    def construct_prompt_helper(self, text: str, history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
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

        history.append(prompt)
        return history
    
    def construct_prompt(self, text: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        if self.max_prompt_length is not None:
            prompt_empty = self.construct_prompt_helper("", history)
            chat_template_length = len(self.tokenizer.apply_tokenizer_chat_template(prompt_empty, tokenize=True))
            
            document_text_tokenized = self.tokenizer.tokenizer.encode(text)
            # remove tokens that exceed max_prompt_length
            if len(document_text_tokenized) > self.max_prompt_length - chat_template_length:
                document_text_tokenized = document_text_tokenized[:self.max_prompt_length - chat_template_length]
                # detokenize the tokens
                # we need to skip special tokens because they are not part of the original text e.g., begin of document token
                document_text_detokenized = self.tokenizer.tokenizer.decode(document_text_tokenized, skip_special_tokens=True)
                # construct the prompt
                prompt_final = self.construct_prompt_helper(document_text_detokenized, history)
                if text[:len(document_text_detokenized)] != document_text_detokenized:
                    raise ValueError("The text was not properly tokenized and detokenized")
                prompt_final = self.tokenizer.apply_tokenizer_chat_template(prompt_final, tokenize=False)
                return prompt_final
        
        prompt_final = self.construct_prompt_helper(text, history)
        prompt_final = self.tokenizer.apply_tokenizer_chat_template(prompt_final, tokenize=False)
        return prompt_final



