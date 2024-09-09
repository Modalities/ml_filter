from typing import Dict, List

from utils.app_config import AppConfig


class PromptGenerator:
    @staticmethod
    def get_prompt(extract, app_config: AppConfig) -> List[Dict[str, str]]:
        chat = [
            {
                "role": "user",
                "content": app_config.fineweb_prompt.format(extract=extract) # Replace {extract} with actual extract
            }
        ]
        return chat
    


