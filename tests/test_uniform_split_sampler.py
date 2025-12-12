import json
import shutil
from collections import Counter
from pathlib import Path

from ml_filter.sampling.uniform_split_sampler import UniformSplitSampler

TESTS_ROOT = Path(__file__).resolve().parent
FIXTURE_INPUT_NAME = "lorem_ipsum_sampling.jsonl"
FIXTURE_INPUT_PATH = TESTS_ROOT / "resources" / "data" / FIXTURE_INPUT_NAME
TARGET_INPUT_FILENAME = "lorem_ipsum_sampling.jsonl"


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_uniform_split_sampler_uses_fixture_distribution(tmp_path):
    input_dir = tmp_path / "input_data"
    input_dir.mkdir()
    output_dir = tmp_path / "output_data"

    fixture_target = input_dir / TARGET_INPUT_FILENAME
    shutil.copy(FIXTURE_INPUT_PATH, fixture_target)

    sampler = UniformSplitSampler(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        validation_fraction=0.2,
        score_column="score",
        random_seed=123,
        max_upsample_factor=1.0,
    )
    sampler.process_all_files()

    train_file = output_dir / "training_set" / f"{TARGET_INPUT_FILENAME.replace('.jsonl', '')}_train.jsonl"
    val_file = output_dir / "validation_set" / f"{TARGET_INPUT_FILENAME.replace('.jsonl', '')}_val.jsonl"

    assert train_file.exists()
    assert val_file.exists()

    train_records = _read_jsonl(train_file)
    val_records = _read_jsonl(val_file)

    assert len(train_records) == 12
    assert len(val_records) == 3

    train_counts = Counter(int(row["score"][0]) for row in train_records)
    val_counts = Counter(int(row["score"][0]) for row in val_records)

    assert train_counts == {0: 4, 1: 4, 2: 4}
    assert val_counts == {0: 1, 1: 1, 2: 1}

    assert all(isinstance(row["score"], list) and len(row["score"]) == 1 for row in train_records + val_records)


def test_uniform_split_sampler_honors_per_label_target(tmp_path):
    input_dir = tmp_path / "input_per_label"
    output_dir = tmp_path / "output_per_label"

    records = []
    for score in range(3):
        for idx in range(5):
            records.append(
                {
                    "id": f"balanced-{score}-{idx}",
                    "score": float(score),
                    "aggregation_type": "majority",
                }
            )

    input_file = input_dir / "balanced_dataset.jsonl"
    _write_jsonl(input_file, records)

    sampler = UniformSplitSampler(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        validation_fraction=0.0,
        score_column="score",
        random_seed=11,
        max_upsample_factor=10.0,
        per_label_target=2,
    )
    sampler.process_all_files()

    train_file = output_dir / "training_set" / "balanced_dataset_train.jsonl"
    train_records = _read_jsonl(train_file)

    train_counts = Counter(int(row["score"][0]) for row in train_records)
    assert train_counts == {0: 2, 1: 2, 2: 2}


def test_uniform_split_sampler_respects_max_upsample_factor(tmp_path):
    input_dir = tmp_path / "input_sparse"
    output_dir = tmp_path / "output_sparse"

    source_counts = {0: 1, 1: 2, 2: 3}
    sparse_records = []
    for score, count in source_counts.items():
        for idx in range(count):
            sparse_records.append(
                {
                    "id": f"sparse-{score}-{idx}",
                    "score": float(score),
                    "aggregation_type": "majority",
                }
            )

    input_file = input_dir / "sparse_dataset.jsonl"
    _write_jsonl(input_file, sparse_records)

    sampler = UniformSplitSampler(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        validation_fraction=0.0,
        score_column="score",
        random_seed=7,
        max_upsample_factor=2.0,
        per_label_target=500,
    )
    sampler.process_all_files()

    train_file = output_dir / "training_set" / "sparse_dataset_train.jsonl"
    val_file = output_dir / "validation_set" / "sparse_dataset_val.jsonl"

    train_records = _read_jsonl(train_file)
    val_records = _read_jsonl(val_file)

    assert not val_records  # validation_fraction=0.0, so no validation records expected

    train_counts = Counter(int(row["score"][0]) for row in train_records)
    assert train_counts == {0: 2, 1: 4, 2: 6}

    for score, available in source_counts.items():
        assert train_counts[score] <= available * sampler.max_upsample_factor
