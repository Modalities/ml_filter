import gc
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from datatrove.data import DocumentsPipeline
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger
from transformers import AutoModel, AutoTokenizer

from ml_filter.annotation.datatrove_jql_annotator import JQLJsonlReader


class StreamingTokenizer(PipelineStep):
    """
    Streaming tokenizer that processes documents individually - no batching needed.
    """

    name = "🔤 Streaming Tokenizer"
    type = "🔤 TOKENIZER"

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int = 8192,
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self._tokenizer = None
        self.doc_count = 0

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.info(f"Initialized streaming tokenizer: {self.tokenizer_name}")
        return self._tokenizer

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        logger.info(f"[StreamingTokenizer][rank={rank}] Processing documents individually")

        for doc in data:
            with self.track_time(unit="doc"):
                self.doc_count += 1

                # Tokenize single document
                encoded = self.tokenizer(
                    doc.text,
                    max_length=self.max_length,
                    padding=False,  # No padding!
                    truncation=True,
                    return_tensors=None,  # Get lists, not tensors
                    return_attention_mask=True,
                )

                # Store unpadded sequence
                doc.metadata["input_ids"] = encoded["input_ids"]
                doc.metadata["attention_mask"] = encoded["attention_mask"]
                doc.metadata["token_count"] = len(encoded["input_ids"])

                self.stat_update("tokens", value=doc.metadata["token_count"])

                if self.doc_count % 1000 == 0:
                    logger.info(f"[StreamingTokenizer][rank={rank}] Processed {self.doc_count} documents")

                yield doc


class UnpackedEmbedder(PipelineStep):
    name = "🔢 Unpacked Embedder"
    type = "🔢 EMBEDDER"

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        max_total_tokens: int = 81920,  # Much higher limit discovered!
        auto_find_limit: bool = False,  # Set to True to find your GPU's limit
    ):
        super().__init__()
        self.model_name = model_name
        self.device_str = device
        self.dtype = dtype
        self.max_total_tokens = max_total_tokens
        self.auto_find_limit = auto_find_limit
        self._model = None
        self._device = None

    def _init_model(self, rank: int):
        if self.device_str:
            self._device = torch.device(f"cuda:{self.device_str}")
        elif torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            cuda_device_id = rank % device_count
            self._device = torch.device(f"cuda:{cuda_device_id}")
        else:
            self._device = torch.device("cpu")

        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            unpad_inputs=True,  # Always True for this implementation
            torch_dtype=self.dtype,
            add_pooling_layer=False,
            use_memory_efficient_attention=True,
        ).to(self._device)
        self._model.eval()

        logger.info(f"Model loaded with unpad_inputs=True, max_tokens={self.max_total_tokens}")

    def _find_max_token_limit(self):
        """Automatically find the maximum token limit for your GPU."""
        if not self.auto_find_limit:
            return self.max_total_tokens

        logger.info("🔍 Auto-finding maximum token limit for your GPU...")

        # Create dummy data
        dummy_input_ids = [0] * 8192  # One full sequence
        dummy_attention_mask = [1] * 8192
        dummy_doc = type(
            "obj",
            (object,),
            {"metadata": {"input_ids": dummy_input_ids, "attention_mask": dummy_attention_mask, "token_count": 8192}},
        )()

        # Start with a working baseline
        working_limit = 8192
        logger.info(f"Testing baseline {working_limit} tokens...")
        try:
            test_docs = [dummy_doc] * 1
            embeddings = self._process_token_batch(test_docs)
            del embeddings
            torch.cuda.empty_cache()
            logger.info(f"✅ Baseline {working_limit} tokens PASSED")
        except Exception as e:
            logger.error(f"❌ Even baseline failed: {e}")
            return working_limit

        # Exponential search: double each time until failure
        current_test = working_limit

        logger.info("🚀 Exponential search for failure point...")
        while True:
            current_test *= 2  # True exponential growth
            num_docs = current_test // 8192
            test_docs = [dummy_doc] * num_docs

            try:
                logger.info(f"Testing {current_test:,} tokens ({num_docs} docs)...")

                with torch.no_grad():
                    embeddings = self._process_token_batch(test_docs)
                    del embeddings
                    torch.cuda.empty_cache()

                logger.info(f"✅ {current_test:,} tokens PASSED")
                working_limit = current_test

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"❌ {current_test:,} tokens FAILED (OOM) - Found failure point!")
                    break
                else:
                    logger.error(f"❌ {current_test:,} tokens FAILED: {e}")
                    break
            except Exception as e:
                logger.error(f"❌ {current_test:,} tokens FAILED: {e}")
                break

            # Safety check - don't go beyond 50M tokens
            if current_test > 50_000_000:
                logger.warning("Reached 50M tokens without failure - stopping search")
                break

        # Binary search between working_limit and current_test (failure point)
        low = working_limit
        high = current_test
        final_limit = working_limit

        logger.info(f"🎯 Binary searching between {low:,} and {high:,} tokens...")

        while high - low > 16384:  # Stop when range is small enough
            mid = ((low + high) // 2 // 8192) * 8192  # Round to doc boundary
            num_docs = mid // 8192
            test_docs = [dummy_doc] * num_docs

            try:
                logger.info(f"Testing {mid:,} tokens ({num_docs} docs)...")

                with torch.no_grad():
                    embeddings = self._process_token_batch(test_docs)
                    del embeddings
                    torch.cuda.empty_cache()

                logger.info(f"✅ {mid:,} tokens PASSED")
                final_limit = mid
                low = mid

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"❌ {mid:,} tokens FAILED (OOM)")
                    high = mid
                    torch.cuda.empty_cache()
                else:
                    logger.error(f"❌ {mid:,} tokens FAILED: {e}")
                    high = mid
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"❌ {mid:,} tokens FAILED: {e}")
                high = mid
                torch.cuda.empty_cache()

        num_full_docs = final_limit // 8192
        logger.info(f"🎯 Maximum token limit found: {final_limit:,} tokens ({num_full_docs} full docs)")
        return final_limit

    def _accumulate_until_token_limit(self, data_iterator):
        """Accumulate documents until token limit is reached."""
        accumulated_docs = []
        current_tokens = 0

        for doc in data_iterator:
            doc_tokens = doc.metadata["token_count"]

            # If adding this doc would exceed limit, yield current batch and start new one
            if current_tokens + doc_tokens > self.max_total_tokens and accumulated_docs:
                yield accumulated_docs
                accumulated_docs = [doc]
                current_tokens = doc_tokens
            else:
                accumulated_docs.append(doc)
                current_tokens += doc_tokens

        # Yield remaining documents
        if accumulated_docs:
            yield accumulated_docs
        """Accumulate documents until token limit is reached."""
        accumulated_docs = []
        current_tokens = 0

        for doc in data_iterator:
            doc_tokens = doc.metadata["token_count"]

            # If adding this doc would exceed limit, yield current batch and start new one
            if current_tokens + doc_tokens > self.max_total_tokens and accumulated_docs:
                yield accumulated_docs
                accumulated_docs = [doc]
                current_tokens = doc_tokens
            else:
                accumulated_docs.append(doc)
                current_tokens += doc_tokens

        # Yield remaining documents
        if accumulated_docs:
            yield accumulated_docs

    def _process_token_batch(self, docs):
        """Process a batch of documents as one concatenated sequence."""
        # Concatenate all sequences
        all_input_ids = []
        all_attention_mask = []
        lengths = []

        for doc in docs:
            input_ids = doc.metadata["input_ids"]
            attention_mask = doc.metadata["attention_mask"]
            all_input_ids.extend(input_ids)
            all_attention_mask.extend(attention_mask)
            lengths.append(len(input_ids))

        actual_total_tokens = len(all_input_ids)

        # Create tensors - single batch with concatenated sequences
        input_ids = torch.tensor([all_input_ids], dtype=torch.long, device=self._device)
        attention_mask = torch.tensor([all_attention_mask], dtype=torch.long, device=self._device)

        # Create token_type_ids of same length (all zeros for single sequence type)
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=self._device)

        logger.info(f"Processing {len(docs)} docs as {actual_total_tokens} concatenated tokens")

        with torch.no_grad():
            # GTE handles sequence reconstruction internally using lengths
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,  # Explicit token_type_ids
                length=lengths,
                unpad_inputs=True,
            )

            # GTE returns embeddings for each original sequence
            embeddings = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)

        return embeddings.cpu()

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if self._model is None:
            self._init_model(rank)

        # Auto-find limit if requested
        if self.auto_find_limit:
            self.max_total_tokens = self._find_max_token_limit()

        logger.info(
            f"[UnpackedEmbedder][rank={rank}] Processing with token-only batching, limit={self.max_total_tokens}"
        )

        total_docs = 0
        total_time = 0.0
        batch_count = 0

        for token_batch in self._accumulate_until_token_limit(data):
            with self.track_time(unit="token_batch"):
                start = time.time()
                batch_count += 1

                try:
                    # Process entire token batch as one operation
                    embeddings = self._process_token_batch(token_batch)

                    # Assign embeddings back to documents
                    for doc, emb in zip(token_batch, embeddings):
                        doc.metadata["embedding"] = emb.tolist()
                        doc.metadata["source_filename"] = Path(doc.metadata.get("file_path", "")).stem

                        # Clean up tokenization data
                        doc.metadata.pop("input_ids", None)
                        doc.metadata.pop("attention_mask", None)
                        doc.metadata.pop("token_count", None)

                        yield doc

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"OOM with {len(token_batch)} docs. Try reducing max_total_tokens.")
                        # Process docs individually as fallback
                        for doc in token_batch:
                            try:
                                single_embedding = self._process_token_batch([doc])
                                doc.metadata["embedding"] = single_embedding[0].tolist()
                                doc.metadata["source_filename"] = Path(doc.metadata.get("file_path", "")).stem
                                doc.metadata.pop("input_ids", None)
                                doc.metadata.pop("attention_mask", None)
                                doc.metadata.pop("token_count", None)
                                yield doc
                            except Exception:
                                logger.error("Failed to process document individually")
                                continue
                    else:
                        raise e

                # Logging
                duration = time.time() - start
                docs_in_batch = len(token_batch)
                total_docs += docs_in_batch
                total_time += duration

                throughput = docs_in_batch / duration if duration > 0 else 0
                avg_throughput = total_docs / total_time if total_time > 0 else 0
                total_tokens = sum(doc.metadata.get("token_count", 0) for doc in token_batch)

                logger.info(
                    f"Token batch {batch_count}: {docs_in_batch} docs, {total_tokens} tokens, "
                    f"{duration:.2f}s → {throughput:.2f} docs/s (avg {avg_throughput:.2f})"
                )

                torch.cuda.empty_cache()
                gc.collect()


def run_token_only_pipeline(config_file_path: Path):
    """Run pipeline with pure token-based batching - no document batch_size needed."""
    from omegaconf import OmegaConf

    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_file_path}")

    cfg = OmegaConf.load(config_file_path)

    pipeline = [
        JQLJsonlReader(cfg.input_dir, Path(cfg.csv_hashmap_path), glob_pattern=cfg.get("glob_pattern", "*.jsonl")),
        StreamingTokenizer(
            tokenizer_name=cfg.get("tokenizer_name", cfg.embedding_model),
            max_length=cfg.get("max_length", 8192),
        ),
        UnpackedEmbedder(
            model_name=cfg.embedding_model,
            device=cfg.get("device", None),
            dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
            max_total_tokens=cfg.get("max_total_tokens", 81920),  # Much higher based on findings
            auto_find_limit=True,
        ),
        # Your output writers
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=cfg.get("tasks", 1),
        workers=cfg.get("workers", -1),
        logging_dir=cfg.output_dir + "/logs",
        skip_completed=cfg.get("skip_completed", True),
    )

    logger.info(f"Starting token-only pipeline: max_total_tokens={cfg.get('max_total_tokens', 81920)}")
    executor.run()
    logger.info("Pipeline completed successfully.")
