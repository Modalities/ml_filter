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
    targets = {lang: round(total_sample_size * pct / 100) for lang, pct in language_distribution.items()}
    logger.info(f"Computed targets (total={total_sample_size}): {targets}")
    return targets


def load_hash_mapping(csv_path: Path) -> dict[str, Path]:
    """
    CSV has columns: file_path,md5
    Returns: {md5: Path(raw_jsonl)}
    """
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
    """Return inverse mapping: {Path(raw_jsonl): md5}.
    """
    inverse: dict[Path, str] = {}
    for md5, p in hash_mapping.items():
        if p in inverse:
            logger.warning(f"Duplicate path in inversion: {p} (old={inverse[p]}, new={md5}) overwriting")
        inverse[p] = md5
    return inverse


def load_jsonl_counts(annotated_base: Path, use_wc: bool = True) -> dict[str, dict[Path, int]]:
    """Traverse annotated_base/<lang>/... and count docs in each JSONL.

    Args:
        annotated_base: Root directory containing per-language subdirectories with JSONL files.
        use_wc: If True (default), attempt fast line counting via external `wc -l` command.
                Falls back to Python iteration on failure.

    Returns:
        {lang: {annotated_file_path: num_docs}}
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
            logger.info(f"Using 'wc -l' to count lines in {len(jsonl_files)} files for language '{lang}'")
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
    """Sample synthetic document IDs of the form <documenthash>_<line_index>.

    Args:
        lang_to_files: { lang: { Path(jsonl): n_docs, ... }, ... }
        targets: { lang: target_count, ... }
        file_to_hash: { Path(jsonl): md5 } mapping (inverse of the CSV md5->path mapping)
        rng: optional random.Random instance for reproducibility

    Returns:
        { lang: [ "<md5>_<line_index>", ... ], ... }
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

        # Step 1: proportional allocation (floating)
        raw_allocs = [target * (n_docs / total_docs) for _, n_docs in files_items]
        allocs = [int(x) for x in raw_allocs]

        # Step 2: distribute remainder by largest fractional part (tie-break random)
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

        # Step 3: cap allocations and reclaim overflow
        remainder = 0
        for i, (_, n_docs) in enumerate(files_items):
            if allocs[i] > n_docs:
                remainder += allocs[i] - n_docs
                allocs[i] = n_docs

        # Step 3b: redistribute reclaimed remainder greedily
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

        # Step 4: sample line indices for each file and synthesize IDs
        for (fpath, n_docs), quota in zip(files_items, allocs):
            if quota <= 0:
                continue
            doc_hash = file_to_hash.get(fpath)
            if doc_hash is None:
                raise KeyError(f"Hash for file {fpath} not found in file_to_hash mapping")
            # sample unique line indices (0-based)
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



# --- Helper to make filter function ---
def make_filter_func_from_ids(doc_ids: list[str], selected_ids: set[str]) -> Callable[[tuple[int, dict[str, np.ndarray]]], bool]:
    def filter_func(item: tuple[int, dict[str, np.ndarray]]) -> bool:
        idx, _ = item
        try:
            return doc_ids[idx] in selected_ids
        except IndexError:
            logging.error(f"Index {idx} not found in doc_ids list")
            return False
    return filter_func


def _filter_ids_for_file(file_path: Path, selected: set[str], inv_map: dict[Path, str]):
    """Return (filtered_ids, target_rows) for a given annotated file.
    filtered_ids: list of selected ids whose md5 prefix matches this file's md5.
    target_rows: {row_index: full_id}
    Assumes doc_id format: <md5>_<row_number>.
    """
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

# --- Tokenized filterer ---
class TokenizedFilterer:
    def __init__(self, tokenized_base: Path, output_folder: Path, hash_mapping: dict[str, Path], inverse_mapping: dict[Path, str], base_file_prefix: Path):
        self._tokenized_base = tokenized_base
        self._output_folder = output_folder
        self._hash_mapping = hash_mapping        # md5 -> raw_path
        self._inverse_mapping = inverse_mapping  # Path(jsonl) -> md5
        self._base_file_prefix = base_file_prefix

    def _prepare_output_path(self, tokenized_file: Path) -> Path:
        tokenized_file_rel = tokenized_file.relative_to(self._tokenized_base)
        output_path = self._output_folder / tokenized_file_rel.with_suffix(".filtered.pbin")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def filter_document(self, annotated_file: Path, selected_ids: set[str]):
        """Filter a tokenized file using synthetic IDs (<hash>_<idx>)."""
        base_hash = self._inverse_mapping.get(annotated_file)
        if not base_hash:
            logger.info(f"Skipping {annotated_file}: no hash mapping found")
            return

        # fast check: do any selected ids reference this hash?
        prefix = base_hash + "_"
        if not any(sid.startswith(prefix) for sid in selected_ids):
            logger.info(f"Skipping {annotated_file}: no selected ids for hash {base_hash}")
            return

        # Build doc_ids list for index alignment (no need to parse JSON; just count lines)
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



# # --- Main pipeline ---
if __name__ == "__main__":
    csv_path = Path("/data/horse/ws/alju972f-tokenization_at_scale/sampling_pipeline_test/hashes.csv")
    annotations_base = Path("/data/horse/ws/alju972f-tokenization_at_scale/sampling_pipeline_test/annotations")
    tokenized_base = Path("/data/horse/ws/alju972f-tokenization_at_scale/sampling_pipeline_test/tokenized")
    output_folder = Path("/data/horse/ws/alju972f-tokenization_at_scale/sampling_pipeline_test/output_filtered")
    base_file_prefix = Path("/data/horse/ws/alju972f-tokenization_at_scale/sampling_pipeline_test/jsonl_files")

    language_distribution = {"als_Latn": 25, "deu_Latn": 75}
    total_sample_size = 10

    # # Load CSV hash mapping
    hash_mapping = load_hash_mapping(csv_path)
    inv_hash_mapping = invert_hash_mapping(hash_mapping)
    # # Count annotated JSONL docs
    lang_to_files = load_jsonl_counts(base_file_prefix)

    # Compute targets
    targets = compute_target_samples(language_distribution, total_sample_size)

    # Sample documents
    selected_doc_ids = sample_documents(lang_to_files, targets, inv_hash_mapping)

    # Filter tokenized files
    filterer = TokenizedFilterer(tokenized_base, output_folder, hash_mapping, inv_hash_mapping, base_file_prefix)
    for lang, ids in selected_doc_ids.items():
        selected_set = set(ids)
        for annotated_file in lang_to_files[lang].keys():
            filterer.filter_document(annotated_file, selected_set)


    # # --- Validation step: re-tokenize and compare ---
    # import json
    import sentencepiece as spm
    # from pathlib import Path

    # # --- Validation step: re-tokenize and compare ---
    # from collections import defaultdict

    # selected_doc_ids = defaultdict(list)

    # # selected_doc_ids["als_Latn"].append("29d82196d55803ab9c792e45b59919bf_273561")
    # # selected_doc_ids["deu_Latn"].append("6f174ddca737f54cea5f34da31e15178_931238")

    sp = spm.SentencePieceProcessor(
        model_file="/data/horse/ws/alju972f-tokenization_at_scale/eurolingua_tokenization/tokenizer/tueken2_tokenizer_model.model"
    )

    for lang, ids in selected_doc_ids.items():
        for data_file in lang_to_files[lang].keys():
            filtered_ids, target_rows = _filter_ids_for_file(data_file, ids, inv_hash_mapping)
            if not filtered_ids:
                continue

            rel = data_file.relative_to(base_file_prefix)
            filtered_file = output_folder / rel.with_suffix(".filtered.pbin")
            source_data = PackedMemMapDatasetBase(filtered_file, sample_key="input_ids", load_index=True)

            # print out which files are being compared
            print(f"Validating {data_file} against {filtered_file} for language {lang}")
            
            selected_lines: list[tuple[int, dict]] = []
            with open(data_file) as f:
                for idx, line in enumerate(f):
                    if idx not in target_rows:
                        continue
                    rec = json.loads(line)
                    selected_lines.append((idx, rec))

            if len(selected_lines) != len(source_data):
                logging.warning(
                    f"Length mismatch for {annotated_file}: filtered_pbin={len(source_data)} selected_lines={len(selected_lines)}"
                )

            selected_lines = random.sample(selected_lines, 3)
            for out_idx, (row_idx, rec) in enumerate(selected_lines):
                if out_idx >= len(source_data):
                    break
                pipeline_tokens = source_data[out_idx]["input_ids"].tolist()
                ref_tokens = sp.encode(rec["text"], out_type=int)
                # First check length mismatch (often most informative / faster)
                if len(pipeline_tokens) != len(ref_tokens):
                    print(
                        f"Length mismatch for line {row_idx}: pipeline_len={len(pipeline_tokens)} ref_len={len(ref_tokens)}"
                    )
                    raise AssertionError(f"❌ Length mismatch for line {row_idx}")
                for i, (p_tok, r_tok) in enumerate(zip(pipeline_tokens, ref_tokens)):
                    if p_tok != r_tok:
                        print(f"Token mismatch at position {i} for line {row_idx}: pipeline={p_tok} ref={r_tok}")
                        break
                raise AssertionError(f"❌ Token mismatch for line {row_idx}")
