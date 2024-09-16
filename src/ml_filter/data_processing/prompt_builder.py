from pathlib import Path
from typing import Dict, List, Optional

import yaml


class PromptBuilder:
    """A class representing a prompt builder."""

    def __init__(self, prompt_path: Path) -> None:
        with open(prompt_path, "r") as file:
            self.prompt_template = yaml.safe_load(file)

    def construct_prompt(self, text: str, history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """Constructs a prompt.

        Args:
            text (str): The text to be used as a placeholder in the prompt.
            history (Optional[List[Dict[str, str]]]): The history of prompts. Defaults to None.

        Returns:
            List[Dict[str, str]]: The prompt that is represtend as a list (history) of messages.
        """

        # TODO: Is this fixed for all models?
        prompt = {"role": "user", "content": self.prompt_template["prompt"].format(placeholder=text)}

        if history is None:
            history = []

        history.append(prompt)
        return history
