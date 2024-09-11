from pathlib import Path
from typing import Dict, List, Optional

import yaml



class PromptBuilder:
    """A class representing a chat template."""

    def __init__(self, prompt_path: Path) -> None:
        with open(prompt_path, 'r') as file:
            self.prompt_template = yaml.safe_load(file)
       

    def construct_prompt(self, text: str, history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        Apply a chat template to the given text.

        Args:
            text (str): The text to be used as a placeholder in the template.

        Returns:
            Dict[str, str]: A dictionary representing a chat conversation.
            The dictionary contains the role of the speaker ("user") and the content
            of their message, which is the template with the placeholder replaced by
            the given text.
        """
        # TODO: Is this fixed for all models?
        prompt = {
            "role": "user",
            "content": self.prompt_template.format(placeholder=text)
        }

        if history is None:
            history = []
        
        history.append(prompt)
        return history
    


