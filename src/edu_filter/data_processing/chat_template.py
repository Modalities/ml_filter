from pathlib import Path
from typing import Dict, List

import yaml



class ChatTemplate:
    """A class representing a chat template."""

    def __init__(self, prompt_path: Path) -> None:
        with open(prompt_path, 'r') as file:
            self.prompt = yaml.safe_load(file)
       

    def apply_template(self, text: str) -> Dict[str, str]:
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
        chat = {
            "role": "user",
            "content": self.prompt.format(placeholder=text)
        }
        
        return chat
    


