from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
from llm_interface.interface.interface import LanguageModelAPI

from utils.helper import PromptGenerator
from utils.prompt_templates import ChatPromptTemplate, Message, Prompt, Role

class DocumentProcessorInterface(ABC):
    @abstractmethod
    def process_document(self, doc: Dict[str, Any], index: int, user_prompt:str) -> Tuple[int, str]:
        pass


class MixtralDocumentProcessor(DocumentProcessorInterface):
    def __init__(self, llm_service: LanguageModelAPI):
        self.llm_service = llm_service

    def process_document(self, doc: Dict[str, Any], index: int, user_prompt:str="") -> Tuple[int, str]:
        element = doc["text"]
        prompt = PromptGenerator.get_mixtral_prompt(element)
        response = self.llm_service.generate(
            prompt=prompt,
            model_name="llm-service-tgi-mistralai-Mixtral-8x22B-Instruct-v0.1",
            decoding={
                "max_tokens": 50000,
                "max_new_tokens": 1000,
                "temperature": 0.001
            }
        )
        return index, response["generated_text"]


class LlamaDocumentProcessor(DocumentProcessorInterface):
    def __init__(self, llm_service: LanguageModelAPI):
        self.llm_service = llm_service

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

    def process_document(self, doc: Dict[str, Any], index: int,user_prompt:str="") -> Tuple[int, str]:
        document_text = doc["text"].replace("|","")
        #doc2 = "Geboren am 22.10.1926 in Hagen Gestorben am 2.4.1977 in Bamberg Hochschule für Theater in FrankfurtMain Wehrdienst und Kriegsgefangenschaft Engangements In Bonn als Schauspieler In Göttingen als Schauspieler In Iserlohn als Schauspieler und Regisseur Im Juni 1958 in Bamberg als Schauspieler und Regisseur Am 1. Februar 1961 Ernennung zum Intendanten des Bamberger ETA Hoffmann Theaters als Gerd Gutbier Nachruf im Spielzeitheft 197778 Geboren 1926 in HagenWestfalen ging er nach abgeschlossener Schulausbildung an die Hochschule für Theater in FrankfurtMain. Noch im April 1944 wurde er zum Wehrdienst einberufen und erhielt nach Entlassung aus der Gefangenschaft sein erstes Engagement in Bonn. Später kam er an das Theater der Jugend Göttingen und nach freiberuflicher Tätigkeit sowie Fortsetzung der Studien war Gerd Gutbier Nachruf Deutsches Bühnenjahrbuch BühnenAngehöriger Genossenschaft Deutscher Hg. 1978 Deutsches Bühnenjahrbuch. d. große Adreßbuch für Bühne Film Funk u. Fernsehen Verl. d. BühnenschriftenVertriebsGes 86. Hier Seite 696 … Toleranz und ein gütiges Wesen zeichneten Gerd Gutbier stets aus. Verständnis für seine Mitmenschen in allen Lagen war eine seiner hervorragenden Eigenschaften…"
        prompt = self.construct_prompt(user_prompt=user_prompt, input_doc=document_text)
        
        response = self.llm_service.generate(
            prompt=prompt,
            model_name="meta-llama-Meta-Llama-3.1-70B-Instruct",
            decoding={
                "max_tokens": 20000,
                "max_new_tokens": 400,
                "stop_sequences": ["#"],
                "temperature": 0.0
            }
        )
        return (index, response["texts"])