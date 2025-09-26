"""Language sampling, filtering, and validation pipeline.

Refactored for clarity, testability, and CLI flexibility.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Tuple, Set

import yaml
import sentencepiece as spm
from pydantic import BaseModel, Field, validator
from ml_filter.src.ml_filter.data_processing.lang_based_sampling.sampling_utils import (
    load_hash_mapping,
    invert_hash_mapping,
    load_jsonl_counts,
    compute_target_samples,
    sample_documents,
    TokenizedFilterer,
)
from modalities.dataloader.dataset import PackedMemMapDatasetBase

EOD_TOKEN_ID = 3  # Sentence end token appended by pipeline, to be ignored in validation

logger = logging.getLogger("language_pipeline")


def setup_logging(level: str) -> None:
    """Configure root logging once."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class PathsConfig(BaseModel):
    tokenized_base: Path
    output_folder: Path
    base_file_prefix: Path
    csv_path: Path
    tokenizer_model: Path

    @validator("tokenized_base", "output_folder", "base_file_prefix", "csv_path", "tokenizer_model", pre=True)
    def _to_path(cls, v):  # type: ignore
        return Path(v)


class SamplingConfig(BaseModel):
    language_distribution: Dict[str, int]
    total_sample_size: int = Field(gt=0)

    @validator("language_distribution")
    def non_empty_distribution(cls, v):  # type: ignore
        if not v:
            raise ValueError("language_distribution cannot be empty")
        return v


class PipelineConfig(BaseModel):
    paths: PathsConfig
    sampling: SamplingConfig


def load_config(path: Path) -> PipelineConfig:
    """Load YAML config into strongly typed PipelineConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)


def run_sampling_and_filter(
    lang: str,
    cfg: PipelineConfig,
    use_wc: bool,
) -> Tuple[Set[str], Dict[str, Dict[Path, int]], dict, dict]:
    """Run sampling for all languages, return selected IDs for requested language and related mappings."""
    paths_cfg = cfg.paths
    sampling_cfg = cfg.sampling
    tokenized_base = paths_cfg.tokenized_base
    output_folder = paths_cfg.output_folder
    base_file_prefix = paths_cfg.base_file_prefix
    csv_path = paths_cfg.csv_path

    language_distribution = sampling_cfg.language_distribution
    total_sample_size = sampling_cfg.total_sample_size
    logger.info(
        f"Sampling config total_sample_size={total_sample_size} distribution={language_distribution}"
    )

    hash_mapping = load_hash_mapping(csv_path)
    inv_hash_mapping = invert_hash_mapping(hash_mapping)
    lang_to_files = load_jsonl_counts(base_file_prefix, use_wc=use_wc)
    targets = compute_target_samples(language_distribution, total_sample_size)
    selected_doc_ids_all = sample_documents(lang_to_files, targets, inv_hash_mapping)
    logger.info(
        "Sampling done: " + ", ".join(f"{k}:{len(v)}" for k, v in selected_doc_ids_all.items())
    )
    if lang not in selected_doc_ids_all:
        raise KeyError(
            f"Language {lang} not present in sampled IDs. Available={list(selected_doc_ids_all.keys())}"
        )
    selected_ids = set(selected_doc_ids_all[lang])
    logger.info(f"Selected {len(selected_ids)} synthetic IDs for {lang}")

    # Filtering
    filterer = TokenizedFilterer(
        tokenized_base,
        output_folder,
        hash_mapping,
        inv_hash_mapping,
        base_file_prefix,
    )
    files_filtered = 0
    for annotated_file in lang_to_files[lang].keys():
        filterer.filter_document(annotated_file, selected_ids)
        files_filtered += 1
    logger.info(f"Filtering complete: files_processed={files_filtered}")
    return selected_ids, lang_to_files, hash_mapping, inv_hash_mapping


def validate_filtered(
    lang: str,
    selected_ids: Set[str],
    lang_to_files: Dict[str, Dict[Path, int]],
    mappings: Tuple[dict, dict],
    cfg: PipelineConfig,
    samples_per_file: int | None = None,
) -> int:
    """Validate filtered tokenized output against SentencePiece encoding.

    Returns number of validated documents.
    """
    hash_mapping, inv_hash_mapping = mappings
    tokenizer_model = cfg.paths.tokenizer_model
    base_file_prefix = cfg.paths.base_file_prefix
    output_folder = cfg.paths.output_folder

    logger.info("Loading SentencePiece model for validation")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    logger.info("Validation phase starting")

    def _filter_ids_for_file(file_path: Path, selected: set[str]):
        base_md5 = inv_hash_mapping.get(file_path)
        if not base_md5:
            return [], {}
        prefix = base_md5 + "_"
        file_ids = [sid for sid in selected if sid.startswith(prefix)]
        if not file_ids:
            return [], {}
        rows: dict[int, str] = {}
        for sid in file_ids:
            try:
                rows[int(sid.rsplit('_', 1)[1])] = sid
            except (ValueError, IndexError):
                logging.warning(f"Malformed doc_id (skipping): {sid}")
                continue
        return file_ids, rows

    validation_docs = 0
    for data_file in lang_to_files[lang].keys():
        filtered_ids, target_rows = _filter_ids_for_file(data_file, selected_ids)
        if not filtered_ids:
            continue
        rel = data_file.relative_to(base_file_prefix)
        filtered_file = output_folder / rel.with_suffix(".filtered.pbin")
        logger.info(
            f"Validating: src_jsonl={data_file} filtered_pbin={filtered_file} ids={len(filtered_ids)}"
        )
        try:
            source_data = PackedMemMapDatasetBase(
                filtered_file, sample_key="input_ids", load_index=True
            )
        except FileNotFoundError:
            logger.error(f"Filtered pbin not found: {filtered_file}")
            continue
        selected_lines: list[tuple[int, dict]] = []
        with open(data_file) as f:
            for idx, line in enumerate(f):
                if idx not in target_rows:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"JSON decode error at line {idx} in {data_file}: {e}"
                    )
                    continue
                selected_lines.append((idx, rec))

        # Optionally subsample validation lines for this file
        if samples_per_file is not None and len(selected_lines) > samples_per_file:
            selected_lines = random.sample(selected_lines, samples_per_file)
        
        #logger the selected lines
        logger.debug(f"Selected lines for validation in {data_file}: {[idx for idx, _ in selected_lines]}")

        for out_idx, (row_idx, rec) in enumerate(selected_lines):
            if out_idx >= len(source_data):
                break
            pipeline_tokens = source_data[out_idx]["input_ids"].tolist()
            ref_tokens = sp.encode(rec["text"], out_type=int)
            had_trailing_eod = False
            if (
                pipeline_tokens
                and pipeline_tokens[-1] == EOD_TOKEN_ID
                and (len(pipeline_tokens) - 1) == len(ref_tokens)
            ):
                had_trailing_eod = True
                logger.debug(f"Trailing EOD token found in line {row_idx} of {data_file}, ignoring for validation")
                compare_pipeline = pipeline_tokens[:-1]
            else:
                compare_pipeline = pipeline_tokens
            base_md5 = inv_hash_mapping.get(data_file)
            synthetic_id = f"{base_md5}_{row_idx}" if base_md5 else f"UNKNOWN_{row_idx}"
            for i, (p_tok, r_tok) in enumerate(zip(compare_pipeline, ref_tokens)):
                if p_tok != r_tok:
                    logger.error(
                        f"Token mismatch file={data_file} line={row_idx} out_idx={out_idx} doc_id={synthetic_id} pos={i} pipeline_tok={p_tok} ref_tok={r_tok}"
                    )
                    raise AssertionError(f"Token mismatch for line {row_idx}")
            if len(compare_pipeline) != len(ref_tokens):
                logger.error(
                    f"Length mismatch file={data_file} line={row_idx} out_idx={out_idx} doc_id={synthetic_id} pipeline_len={len(compare_pipeline)} ref_len={len(ref_tokens)} had_trailing_eod={had_trailing_eod} original_pipeline_len={len(pipeline_tokens)}"
                )
                raise AssertionError(f"Length mismatch for line {row_idx}")
            validation_docs += 1
        logger.info(
            f"Validated file {data_file} ok (docs_validated={len(selected_lines)})"
        )
    logger.info(f"Validation complete: total_validated_docs={validation_docs}")
    return validation_docs


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run language sampling/filtering/validation pipeline"
    )
    p.add_argument("lang", help="Language code to process (must exist in distribution)")
    p.add_argument(
        "--config", "-c", default=os.environ.get("PIPELINE_CONFIG", "config.yaml"),
        help="Path to YAML config (default env PIPELINE_CONFIG or config.yaml)",
    )
    p.add_argument(
        "--log-level", default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging level (default from LOG_LEVEL env or INFO)",
    )
    p.add_argument(
        "--skip-validation", action="store_true", help="Skip token-level validation phase"
    )
    p.add_argument(
        "--disable-wc", action="store_true", help="Disable fast wc -l counting"
    )
    p.add_argument(
        "--validation-samples-per-file", type=int, default=5,
        help="Maximum number of documents to validate per file (random sample). If omitted, validate all."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    setup_logging(args.log_level)
    logger.info(f"Starting pipeline lang={args.lang}")
    cfg = load_config(Path(args.config))
    selected_ids, lang_to_files, hash_mapping, inv_hash_mapping = run_sampling_and_filter(
        args.lang, cfg, use_wc=not args.disable_wc
    )
    if args.skip_validation:
        logger.info("Validation skipped by flag")
    else:
        validate_filtered(
            args.lang,
            selected_ids,
            lang_to_files,
            (hash_mapping, inv_hash_mapping),
            cfg,
            samples_per_file=args.validation_samples_per_file,
        )
    logger.info(f"Job for {args.lang} completed successfully ✅")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
