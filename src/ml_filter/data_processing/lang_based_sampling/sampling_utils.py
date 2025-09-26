import json
import csv
import logging
from pathlib import Path
from collections import defaultdict
from typing import Callable
import random
import numpy as np
import subprocess
from typing import Dict, List, Optional
from modalities.dataloader.filter_packed_data import filter_dataset
from modalities.dataloader.dataset import PackedMemMapDatasetBase


logger = logging.getLogger(__name__)

def compute_target_samples(language_distribution: dict[str, int], total_sample_size: int) -> dict[str, int]:
    """Return per-language target counts from percentage distribution.

    language_distribution: mapping language -> percent (0-100)
    total_sample_size: total desired documents across all languages
    Returns: mapping language -> integer target count (rounded)
    """
    targets = {lang: round(total_sample_size * pct / 100) for lang, pct in language_distribution.items()}
    logger.info(f"Computed targets (total={total_sample_size}): {targets}")
    return targets


def load_hash_mapping(csv_path: Path) -> dict[str, Path]:
    """Load md5 -> raw file path mapping from a CSV with columns md5,file_path."""
    mapping = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        n = 0
        for row in reader:
            mapping[row["md5"]] = Path(row["file_path"])
            n += 1
    logger.info(f"Loaded hash mapping entries={n} from {csv_path}")
    return mapping


def invert_hash_mapping(hash_mapping: dict[str, Path]) -> dict[Path, str]:
    """Invert md5->path mapping to path->md5, warning on duplicates."""
    inverse: dict[Path, str] = {}
    for md5, p in hash_mapping.items():
        if p in inverse:
            logger.warning(f"Duplicate path in inversion: {p} (old={inverse[p]}, new={md5}) overwriting")
        inverse[p] = md5
    return inverse


def load_jsonl_counts(annotated_base: Path, use_wc: bool = True) -> dict[str, dict[Path, int]]:
    """Return nested mapping lang -> jsonl file -> line count.

    Uses `wc -l` when use_wc is True for speed, falling back to Python counting on failure.
    annotated_base: root containing per-language subdirectories.
    use_wc: attempt shell wc -l acceleration.
    """
    lang_to_files: dict[str, dict[Path, int]] = defaultdict(dict)

    def count_lines_py(path: Path) -> int:
        with open(path, "r") as f:
            return sum(1 for _ in f)

    for lang_dir in annotated_base.iterdir():
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        jsonl_files = list(lang_dir.rglob("*.jsonl"))
        if not jsonl_files:
            logger.debug(f"No JSONL files found under {lang_dir}")
            continue
        if use_wc:
            logger.info(f"Using wc -l for {len(jsonl_files)} files lang={lang}")
            try:
                cmd = ["wc", "-l", *[str(p) for p in jsonl_files]]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                for line in result.stdout.strip().splitlines():
                    line = line.strip()
                    if not line or line.startswith("total"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    try:
                        count = int(parts[0])
                    except ValueError:
                        continue
                    fname = " ".join(parts[1:])
                    fpath = Path(fname)
                    if fpath in jsonl_files:
                        lang_to_files[lang][fpath] = count

                for p in jsonl_files:
                    if p not in lang_to_files[lang]:
                        lang_to_files[lang][p] = count_lines_py(p)
            except Exception as e:
                logger.warning(f"wc -l failed for lang={lang} falling back to Python counting: {e}")
                for p in jsonl_files:
                    lang_to_files[lang][p] = count_lines_py(p)
        else:
            for p in jsonl_files:
                lang_to_files[lang][p] = count_lines_py(p)

    total_files = sum(len(files) for files in lang_to_files.values())
    logger.info(
        f"Counted JSONL files total_files={total_files} details=" +
        ", ".join(f"{lg}:{len(files)}" for lg, files in lang_to_files.items())
    )
    return lang_to_files


def sample_documents(
    lang_to_files: Dict[str, Dict[Path, int]],
    targets: Dict[str, int],
    file_to_hash: dict[Path, str],
    rng: Optional[random.Random] = None
) -> Dict[str, List[str]]:
    """Sample document line indices proportionally per file to meet per-language targets.

    lang_to_files: lang -> file -> line count
    targets: lang -> desired sample size
    file_to_hash: file path -> md5 hash used in synthetic id prefix
    rng: optional random.Random for deterministic tests
    Returns: lang -> list of synthetic ids <md5>_<line_index>
    """
    if rng is None:
        rng = random.Random()

    selected_doc_ids: Dict[str, List[str]] = defaultdict(list)

    for lang, target in targets.items():
        files_items = list(lang_to_files.get(lang, {}).items())
        if not files_items:
            raise ValueError(f"No files found for language '{lang}'")

        total_docs = sum(n_docs for _, n_docs in files_items)
        if total_docs < target:
            raise ValueError(f"Not enough documents for {lang}: target={target}, available={total_docs}")
        if target == 0:
            continue

        logger.info(f"Sampling lang={lang} target={target} files={len(files_items)} total_docs={total_docs}")

        raw_allocs = [target * (n_docs / total_docs) for _, n_docs in files_items]
        allocs = [int(x) for x in raw_allocs]
        remainder = target - sum(allocs)
        if remainder > 0:
            frac_info = [(i, raw_allocs[i] - allocs[i]) for i in range(len(files_items))]
            rng.shuffle(frac_info)
            frac_info.sort(key=lambda x: x[1], reverse=True)
            for i, _ in frac_info:
                if remainder <= 0:
                    break
                if allocs[i] < files_items[i][1]:
                    allocs[i] += 1
                    remainder -= 1
        logger.debug(f"ALLOC_INITIAL lang={lang} allocs={allocs}")
        remainder = 0
        for i, (_, n_docs) in enumerate(files_items):
            if allocs[i] > n_docs:
                remainder += allocs[i] - n_docs
                allocs[i] = n_docs
        while remainder > 0:
            candidates = [i for i, (_, n_docs) in enumerate(files_items) if allocs[i] < n_docs]
            if not candidates:
                raise RuntimeError(f"Could not redistribute {remainder} samples for {lang}")
            rng.shuffle(candidates)
            candidates.sort(key=lambda i: (files_items[i][1] - allocs[i]), reverse=True)
            for i in candidates:
                if remainder <= 0:
                    break
                allocs[i] += 1
                remainder -= 1
        logger.debug(f"ALLOC_FINAL lang={lang} allocs={allocs}")
        for (fpath, n_docs), quota in zip(files_items, allocs):
            if quota <= 0:
                continue
            doc_hash = file_to_hash.get(fpath)
            if doc_hash is None:
                raise KeyError(f"Hash for file {fpath} not found in file_to_hash mapping")
            if quota > n_docs:
                raise ValueError(f"Quota {quota} exceeds available lines {n_docs} in {fpath}")
            if quota == n_docs:
                indices = list(range(n_docs))
            else:
                indices = rng.sample(range(n_docs), quota)
            ids = [f"{doc_hash}_{idx}" for idx in indices]
            selected_doc_ids[lang].extend(ids)
            logger.debug(
                f"SAMPLED lang={lang} file={fpath.name} hash={doc_hash} quota={quota} first_ids={ids[:5]}"
            )
        logger.info(f"Completed sampling lang={lang} selected={len(selected_doc_ids[lang])}")

    return selected_doc_ids


def make_filter_func_from_ids(doc_ids: list[str], selected_ids: set[str]) -> Callable[[tuple[int, dict[str, np.ndarray]]], bool]:
    """Build predicate for filtering PackedMemMap items by synthetic id membership."""
    def filter_func(item: tuple[int, dict[str, np.ndarray]]) -> bool:
        idx, _ = item
        try:
            return doc_ids[idx] in selected_ids
        except IndexError:
            logging.error(f"Index {idx} not found in doc_ids list")
            return False
    return filter_func


def _filter_ids_for_file(file_path: Path, selected: set[str], inv_map: dict[Path, str]):
    """Return (ids_for_file, index_map) for one annotated file based on selected ids."""
    base_md5 = inv_map.get(file_path)
    if not base_md5:
        raise ValueError(f"File path {file_path} not found in inverse hash mapping")
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
    logger.debug(f"FILTER_IDS file={file_path} hash={base_md5} count={len(file_ids)}")
    return file_ids, rows


class TokenizedFilterer:
    """Filter tokenized packed dataset files based on selected synthetic ids.

    Maps annotated jsonl files (line-based) to corresponding tokenized .pbin files
    and writes filtered copies preserving only selected document indices.
    """
    def __init__(self, tokenized_base: Path, output_folder: Path, hash_mapping: dict[str, Path], inverse_mapping: dict[Path, str], base_file_prefix: Path):
        """Create filterer.

        tokenized_base: root directory of tokenized .pbin files
        output_folder: destination root for filtered outputs
        hash_mapping: md5 -> raw path mapping (from CSV)
        inverse_mapping: raw path -> md5 mapping
        base_file_prefix: common prefix directory of raw paths for relative resolution
        """
        self._tokenized_base = tokenized_base
        self._output_folder = output_folder
        self._hash_mapping = hash_mapping
        self._inverse_mapping = inverse_mapping
        self._base_file_prefix = base_file_prefix

    def _prepare_output_path(self, tokenized_file: Path) -> Path:
        """Return output path for a tokenized file ensuring parent directories exist."""
        tokenized_file_rel = tokenized_file.relative_to(self._tokenized_base)
        output_path = self._output_folder / tokenized_file_rel.with_suffix(".filtered.pbin")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def filter_document(self, annotated_file: Path, selected_ids: set[str]):
        """Filter one annotated jsonl file's corresponding tokenized file using selected ids.

        annotated_file: path to the annotated jsonl
        selected_ids: full set of synthetic ids to retain (across languages)
        """
        base_hash = self._inverse_mapping.get(annotated_file)
        if not base_hash:
            logger.info(f"Skipping {annotated_file}: no hash mapping found")
            return

        prefix = base_hash + "_"
        if not any(sid.startswith(prefix) for sid in selected_ids):
            logger.info(f"Skipping {annotated_file}: no selected ids for hash {base_hash}")
            return
        with open(annotated_file, "r") as f:
            doc_ids = [f"{base_hash}_{i}" for i, _ in enumerate(f)]
        logger.debug(
            f"FILTER file={annotated_file.name} hash={base_hash} total_lines={len(doc_ids)} selected_in_hash={sum(1 for i in selected_ids if i.startswith(prefix))}"
        )

        if base_hash not in self._hash_mapping:
            raise ValueError(f"Base hash {base_hash} not found in CSV hash mapping")

        raw_path = self._hash_mapping[base_hash]
        try:
            rel = raw_path.relative_to(self._base_file_prefix)
            tokenized_file = (self._tokenized_base / rel).with_suffix(".pbin")
        except ValueError:
            raise ValueError(f"Raw path {raw_path} is not under base prefix {self._base_file_prefix}")
        output_path = self._prepare_output_path(tokenized_file)
        filter_func = make_filter_func_from_ids(doc_ids, selected_ids)
        logger.info(f"Filtering hash={base_hash} src={tokenized_file} -> dst={output_path}")
        filter_dataset(src_path=tokenized_file, dst_path=output_path, filter_func=filter_func)
        logger.info(f"Finished filtering hash={base_hash} output={output_path}")