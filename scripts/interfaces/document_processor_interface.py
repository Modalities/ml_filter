from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from llm_interface.interface.interface import LanguageModelAPI
import re
from utils import app_config
from utils.app_config import AppConfig
from utils.helper import PromptGenerator
from utils.prompt_templates import ChatPromptTemplate, Message, Prompt, Role
import sys

class DocumentProcessorInterface(ABC):
    @abstractmethod
    def process(self, doc: Dict[str, Any], index: int) -> Tuple[int, str]:
        ...

    @abstractmethod
    def remove_special_strings(self,input_string: str)->str:
        ...


class LmsLlamaDocumentProcessor(DocumentProcessorInterface):
    def __init__(self, llm_service: LanguageModelAPI, app_config: AppConfig):
        self.llm_service = llm_service
        self.app_config = app_config
        

    def format_prompt(self,input_prompt, replacement_string):
        return input_prompt.format(**replacement_string)

    def construct_prompt(self, input_doc) -> Prompt:
        user_instruction = self.format_prompt(self.app_config.fineweb_prompt, {"extract": input_doc})
        template = [Message(role=Role.USER, text=user_instruction)]
        
        chat_prompt = ChatPromptTemplate(
            instruction="You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user.",
            few_shots=[],  # Empty few_shots
            template=template
        )
        
        prompt = Prompt(
            placeholder_values={},
            history=[],
            prompt_template=chat_prompt
        )
        return prompt    
    
    def remove_special_strings(self,input_string: str)->str:
        """
        Removes specific characters or strings from the input string.

        Args:
            input_string (str): The original string.
            strings_to_remove (list): A list of characters or strings to remove from the input string.

        Returns:
            str: The formatted string with specified characters or strings removed.
        """
        strings_to_remove = ["<", "|", "begin_of_text", ">", "<", "|", "start_header_id", "|", ">", "<", "|", "end_header_id", "|", ">","[","…","]","„","“","–","‘",";",":","/","-",'"',"{","}","\\","”",")","(","www.","%","→","!","*","#","’","..",".com"]
        # Remove new lines
        input_string = input_string.replace('\n', '').replace('\r', ' ')
        for string in strings_to_remove:
            input_string = input_string.replace(string, ' ')

        # Remove extra spaces
        input_string = re.sub(r'\s+', '', input_string)
        
        return input_string

    def process(self, doc: Dict[str, Any], index: int) -> Tuple[int, str]:
        document_text = self.remove_special_strings(doc["text"])
        
        prompt = self.construct_prompt(user_prompt=self.app_config.fineweb_prompt, input_doc=document_text)
        
        response = self.llm_service.generate(
            prompt=prompt,
            model_name="meta-llama-Meta-Llama-3.1-70B-Instruct",
            decoding={
                "max_tokens": 120000,
                "max_new_tokens": 500,
                "stop_sequences": [],
                "temperature": 0.0
            }
        )
        return (index, response["texts"])
        #else:
        #    return (index, "Document could not be processed score:0")



class DocumentProcessor(DocumentProcessorInterface):
    def __init__(self, llm_service: LanguageModelAPI, app_config: AppConfig):
        self.llm_service = llm_service
        self.app_config = app_config


    def remove_special_strings(self,input_string: str, strings_to_remove: List[str])->str:
        """
        Removes specific characters or strings from the input string.

        Args:
            input_string (str): The original string.
            strings_to_remove (list): A list of characters or strings to remove from the input string.

        Returns:
            str: The formatted string with specified characters or strings removed.
        """
       
        input_string = input_string.replace('\n', '').replace('\r', ' ')
        for string in list(strings_to_remove):
            input_string = input_string.replace(string, ' ')

        # Remove extra spaces
        input_string = re.sub(r'\s+', '', input_string)
        
        return input_string  

    def process(self, doc: Dict[str, Any], index: int) -> Tuple[int, str]:
        document = self.remove_special_strings(doc["text"],self.app_config.strings_to_remove)
        prompt = PromptGenerator.get_prompt(document , app_config=self.app_config)

        response = self.llm_service.generate(prompt=prompt)
        if response is None:
            # Handle the case where response is None
            return index, "Response is None. score:-1"
        
        elif "generated_text" not in response:
            # For cases where we cant process the document write score: -1 along with the actual error in final json
            return index, f"{response} score:-1 "
        else:
            return index, response["generated_text"]