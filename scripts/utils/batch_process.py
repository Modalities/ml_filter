from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from llm_interface.interface.interface import LanguageModelAPI
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from interfaces.document_processor_interface import DocumentProcessorInterface, MixtralDocumentProcessor
from utils.app_config import AppConfig


class BatchProcessor:
    def __init__(self,  document_processor: DocumentProcessorInterface, max_words: int = 30000):
        self.document_processor = document_processor
        self.max_length = max_words
        self.lock = Lock()

    def create_batches(self, data: List[Dict[str, Any]]) -> List[List[Tuple[int, Dict[str, Any]]]]:
        batches = []
        current_batch = []
        current_batch_word_count = 0

        for index, doc in enumerate(data):
            doc_word_count = len(doc["text"].split(" "))
            if doc_word_count > self.max_length:
                print(f"Document {index} too large, to fit in context length .... ")
                #batches.append([(index, {"text": "Document too large to fit"})])
                continue

            if current_batch_word_count + doc_word_count > self.max_length:
                batches.append(current_batch)
                current_batch = []
                current_batch_word_count = 0

            current_batch.append((index, doc))
            current_batch_word_count += doc_word_count

        if current_batch:
            batches.append(current_batch)

        return batches

    def process_batch(self, batch: List[Tuple[int, Dict[str, Any]]], results: List[Tuple[int, str]], pbar: tqdm,user_prompt: str=""):
        local_results = []
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(self.process_document, doc, user_prompt, index): index
                for index, doc in batch
            }
            #FIXME should i be using as_completed(futures) ?
            for future in futures:
                local_results.append(future.result())

                #Updating for each document instead of each batch
                with self.lock:
                    results.extend(local_results)
                    pbar.update(1)
        
        # with self.lock:
        #     results.extend(local_results)
        #     pbar.update(len(batch))

    def process_document(self, doc: Dict[str, Any], user_prompt: str, index: int) -> Tuple[int, str]:
        return self.document_processor.process(doc=doc,index=index,user_prompt=user_prompt)
