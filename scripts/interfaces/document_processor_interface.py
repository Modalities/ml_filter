from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
from llm_interface.interface.interface import LanguageModelAPI
import re
from utils import app_config
from utils.app_config import AppConfig
from utils.helper import PromptGenerator
from utils.prompt_templates import ChatPromptTemplate, Message, Prompt, Role
import sys

class DocumentProcessorInterface(ABC):
    @abstractmethod
    def process(self, doc: Dict[str, Any], index: int, user_prompt:str) -> Tuple[int, str]:
        ...

    @abstractmethod
    def remove_special_strings(self,input_string: str)->str:
        ...

class MixtralDocumentProcessor(DocumentProcessorInterface):
    def __init__(self, llm_service: LanguageModelAPI, app_config: AppConfig):
        self.llm_service = llm_service
        self.app_config = app_config



    def remove_special_strings(self,input_string: str)->str:
        """
        Removes specific characters or strings from the input string.

        Args:
            input_string (str): The original string.
            strings_to_remove (list): A list of characters or strings to remove from the input string.

        Returns:
            str: The formatted string with specified characters or strings removed.
        """
        strings_to_remove = ["[INST]","</s>","<s>",">","[","…","]","„","“","–","‘",";",":","/","-",'"',"{","}","\\","”",")","(","www.","%","→","!","*","#","’","..",".com"]
        # Remove new lines
        input_string = input_string.replace('\n', '').replace('\r', ' ')
        for string in strings_to_remove:
            input_string = input_string.replace(string, ' ')

        # Remove extra spaces
        input_string = ' '.join(input_string.split())
        
        return input_string    

    def process(self, doc: Dict[str, Any], index: int, user_prompt:str="") -> Tuple[int, str]:
        element = self.remove_special_strings(doc["text"])
        prompt = PromptGenerator.get_prompt(element)
        #llm-service-tgi-mistralai-Mixtral-8x22B-Instruct-v0.1
        response = self.llm_service.generate(
            prompt=prompt,
            model_name="mistralai/Mixtral-8x22B-Instruct-v0.1",
            decoding={
                "max_tokens": 60000,
                "max_new_tokens": 500,
                "temperature": 0.001
            }
        )
        if response is None:
            # Handle the case where response is None
            return index, "Response is None. score:-1"
        
        elif "generated_text" not in response:
            # For cases where we cant process the document write score: -1 along with the actual error in final json
            return index, f"{response} score:-1 "
        else:
            return index, response["generated_text"]


class LlamaDocumentProcessor(DocumentProcessorInterface):
    def __init__(self, llm_service: LanguageModelAPI, app_config: AppConfig):
        self.llm_service = llm_service
        self.app_config = app_config


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
        input_string = ' '.join(input_string.split())
        
        return input_string  

    def process(self, doc: Dict[str, Any], index: int,user_prompt:str="") -> Tuple[int, str]:
        element = self.remove_special_strings(doc["text"])
        prompt = PromptGenerator.get_prompt(element)
        #llm-service-tgi-mistralai-Mixtral-8x22B-Instruct-v0.1

        print(f"The app_config is {self.app_config}")
        sys.exit(0)
        response = self.llm_service.generate(
            prompt=prompt,
            model_name=self.app_config.model_name,
            decoding={
                "max_tokens": self.app_config.max_tokens,
                "max_new_tokens": self.app_config.max_new_tokens,
                "temperature": self.app_config.temperature
            }
        )
        if response is None:
            # Handle the case where response is None
            return index, "Response is None. score:-1"
        
        elif "generated_text" not in response:
            # For cases where we cant process the document write score: -1 along with the actual error in final json
            return index, f"{response} score:-1 "
        else:
            return index, response["generated_text"]


class LmsLlamaDocumentProcessor(DocumentProcessorInterface):
    def __init__(self, llm_service: LanguageModelAPI, app_config: AppConfig):
        self.llm_service = llm_service
        self.app_config = app_config
        

    def format_prompt(self,input_prompt, replacement_string):
        return input_prompt.format(**replacement_string)

    def construct_prompt(self,user_prompt: str, input_doc) -> Prompt:
        user_instruction = self.format_prompt(user_prompt, {"extract": input_doc})
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
        input_string = ' '.join(input_string.split())
        
        return input_string

    def process(self, doc: Dict[str, Any], index: int, user_prompt:str="") -> Tuple[int, str]:
        document_text = self.remove_special_strings(doc["text"])
        #doc2 = "Geboren am 22.10.1926 in Hagen Gestorben am 2.4.1977 in Bamberg Hochschule für Theater in FrankfurtMain Wehrdienst und Kriegsgefangenschaft Engangements In Bonn als Schauspieler In Göttingen als Schauspieler In Iserlohn als Schauspieler und Regisseur Im Juni 1958 in Bamberg als Schauspieler und Regisseur Am 1. Februar 1961 Ernennung zum Intendanten des Bamberger ETA Hoffmann Theaters als Gerd Gutbier Nachruf im Spielzeitheft 197778 Geboren 1926 in HagenWestfalen ging er nach abgeschlossener Schulausbildung an die Hochschule für Theater in FrankfurtMain. Noch im April 1944 wurde er zum Wehrdienst einberufen und erhielt nach Entlassung aus der Gefangenschaft sein erstes Engagement in Bonn. Später kam er an das Theater der Jugend Göttingen und nach freiberuflicher Tätigkeit sowie Fortsetzung der Studien war Gerd Gutbier Nachruf Deutsches Bühnenjahrbuch BühnenAngehöriger Genossenschaft Deutscher Hg. 1978 Deutsches Bühnenjahrbuch. d. große Adreßbuch für Bühne Film Funk u. Fernsehen Verl. d. BühnenschriftenVertriebsGes 86. Hier Seite 696 … Toleranz und ein gütiges Wesen zeichneten Gerd Gutbier stets aus. Verständnis für seine Mitmenschen in allen Lagen war eine seiner hervorragenden Eigenschaften…"
        prompt = self.construct_prompt(user_prompt=user_prompt, input_doc=document_text)
        
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