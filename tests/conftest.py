import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf

from ml_filter.translation.translate import DeepLClient, OpenAIClient, Translator


@pytest.fixture
def tmp_jsonl_directory_with_consistent_and_inconsisten_files(tmp_path):
    """Fixture to create a temporary directory with JSONL files containing content."""
    # Define consistent JSONL file paths and content
    consistent_files = [
        tmp_path / "file_1_common_suffix.jsonl",
        tmp_path / "file_2_common_suffix.jsonl",
        tmp_path / "file_3_common_suffix.jsonl",
    ]
    consistent_content = [{"id": 1, "text": "Document 1"}, {"id": 2, "text": "Document 2"}]
    unique_file_name_stems = set(["common_suffix", "different_suffix"])

    file_name_keep_idx = [2, 3]

    # Write consistent content to JSONL files
    for file in consistent_files:
        with file.open("w") as f:
            for doc in consistent_content:
                f.write(f"{json.dumps(doc)}\n")

    # Define an inconsistent JSONL file path and content
    inconsistent_file = tmp_path / "file_4_different_suffix.jsonl"
    inconsistent_content = [{"id": 3, "text": "Inconsistent Document"}]

    # Write inconsistent content to the JSONL file
    with inconsistent_file.open("w") as f:
        for doc in inconsistent_content:
            f.write(f"{json.dumps(doc)}\n")

    return tmp_path, consistent_files, inconsistent_file, unique_file_name_stems, file_name_keep_idx


@pytest.fixture
def merge_files_tmp_directory(tmp_path: Path):
    # Create temporary JSONL files
    file1 = tmp_path / "data_part1_001_temp_file.jsonl"
    file2 = tmp_path / "data_part1_002_temp_file.jsonl"
    file_name_keep_idx = [3, 4]

    # Content for file1
    content1 = [{"id": "3", "value": "third"}, {"id": "1", "value": "first"}]

    # Content for file2
    content2 = [{"id": "2", "value": "second"}, {"id": "10", "value": "tenth"}]

    # Write to file1
    with open(file1, "w") as f:
        for doc in content1:
            f.write(json.dumps(doc) + "\n")

    # Write to file2
    with open(file2, "w") as f:
        for doc in content2:
            f.write(json.dumps(doc) + "\n")

    return tmp_path, file_name_keep_idx


@pytest.fixture
def deepl_translator():
    deepl_client = DeepLClient(api_key="fake_key", ignore_tag_text="notranslate")
    return Translator(client=deepl_client)


@pytest.fixture
def openai_translator():
    openai_client = OpenAIClient(api_key="fake_key", ignore_tag_text="notranslate")
    return Translator(client=openai_client)


@pytest.fixture
def create_input_yaml():
    """Creates a sample YAML file for testing."""
    data = {"prompt": "Translate the text to {##TARGET_LANGUAGE##}."}
    with TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "input.yaml"
        with open(input_file, "w") as file:
            yaml.safe_dump(data, file)
        yield input_file


@pytest.fixture
def mock_translate_text():
    """Fixture to mock the translate_text method."""

    def translate_mock(text, source_language, target_language):
        return f"{text} translated to {target_language}"

    return translate_mock


@pytest.fixture
def temporary_jsonl_file(tmp_path):
    """Fixture to create a temporary JSONL input file."""
    file_path = tmp_path / "input.jsonl"
    documents = [
        {"text": "Hello world!", "id": 1},
        {"text": "How are you?", "id": 2},
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
    return file_path


@pytest.fixture
def tmp_input_file(tmp_path):
    """
    Creates a temporary input JSONL file for testing.
    """
    input_file = tmp_path / "test_input.jsonl"
    content = [
        {"text": "This is a test document."},
        {"text": "Another test."},
        {"text": "Yet another example of a document with more words."},
        {"text": "Short one."},
    ]
    with input_file.open("w", encoding="utf-8") as f:
        for entry in content:
            f.write(json.dumps(entry) + "\n")
    return input_file


@pytest.fixture
def output_folder(tmp_path):
    """Fixture to create a temporary output folder."""
    output_path = tmp_path / "output"
    output_path.mkdir()
    return output_path


@pytest.fixture
def tmp_output_file(tmp_path):
    """
    Provides a temporary output file path.
    """
    return tmp_path / "test_output.json"


@pytest.fixture
def tmp_jsonl_directory(tmp_path: Path):
    """Fixture to create a temporary directory with JSONL files."""
    directory = tmp_path / "jsonl_files"
    directory.mkdir()
    for i in range(3):
        file = directory / f"file_{i}.jsonl"
        with open(file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": f"This is file {i}"}) + "\n")
    return directory


@pytest.fixture
def tmp_nested_jsonl_directory(tmp_path: Path):
    """Fixture to create a temporary nested directory structure with JSONL files."""
    root_directory = tmp_path / "nested_jsonl_files"
    root_directory.mkdir()

    expected_word_counts = {
        str(root_directory / "root_file_0.jsonl"): 5,
        str(root_directory / "root_file_1.jsonl"): 5,
        str(root_directory / "sub_folder" / "sub_file_0.jsonl"): 3,
        str(root_directory / "sub_folder" / "sub_file_1.jsonl"): 3,
    }

    # Create files in the root directory
    for i in range(2):
        file = root_directory / f"root_file_{i}.jsonl"
        with open(file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": f"This is root file {i}"}) + "\n")

    # Create a nested subdirectory
    sub_directory = root_directory / "sub_folder"
    sub_directory.mkdir()
    for i in range(2):
        file = sub_directory / f"sub_file_{i}.jsonl"
        with open(file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": f"Sub file {i}"}) + "\n")

    return root_directory, expected_word_counts


@pytest.fixture
def temp_output_dir(tmp_path):
    """Creates a temporary output directory for testing."""
    return tmp_path / "output"


@pytest.fixture
def config_file(temp_output_dir):
    """Creates a real configuration file for testing."""
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "training": {
            "output_dir_path": str(temp_output_dir),
            "batch_size": 2,
            "epochs": 1,
            "weight_decay": 0.01,
            "learning_rate": 5e-5,
            "save_strategy": "epoch",
            "logging_steps": 10,
            "logging_dir_path": str(temp_output_dir / "logs"),
            "metric_for_best_model": "accuracy",
            "use_bf16": False,
            "greater_is_better": True,
            "is_regression": True,
            "eval_strategy": "steps",
            "dataloader_num_workers": 1,
        },
        "model": {
            "name": "facebookai/xlm-roberta-base",
            "freeze_base_model_parameters": True,
            "is_regression": True,
        },
        "data": {
            "text_column": "text",
            "label_column": "labels",
            "document_id_column": "id",
            "train_file_path": str(temp_output_dir / "train.jsonl"),
            "train_file_split": "train",
            "val_file_path": str(temp_output_dir / "val.jsonl"),
            "val_file_split": "train",
            "test_file_path": str(temp_output_dir / "test.jsonl"),
            "test_file_split": "train",
            "num_tasks": 3,
            "task_names": ["edu", "toxicity", "adult"],
            "num_targets_per_task": [2, 3, 4],
            "num_processes": 2,
        },
        "tokenizer": {
            "pretrained_model_name_or_path": "facebookai/xlm-roberta-base",
            "add_generation_prompt": False,
            "max_length": 128,
            "padding": "max_length",
            "truncation": True,
        },
    }

    config_path = temp_output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    return config_path


@pytest.fixture
def example_df():
    data = {
        "Annotator": ["annotator1", "annotator2", "annotator3"],
        "metric1": [0.8, 0.9, 0.85],
        "metric2": [0.2, 0.1, 0.15],
        "Invalid": [2, 3, 4],
        "lang": ["en", "en", "en"],
        "Filepath": ["path1", "path2", "path3"],
        "CM": [
            {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}},
            {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}},
            {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}},
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def example_aggregated_metrics_df():
    data = {
        "metric1": {
            "annotator1": 0.8,
            "annotator2": 0.4,
            "annotator3": 0.5,
        },
        "metric2": {
            "annotator1": 0.2,
            "annotator2": 0.3,
            "annotator3": 0.5,
        },
    }
    return pd.DataFrame(data)


@pytest.fixture
def example_top_n_annotators():
    return {
        1: {
            "metric1": {"annotator1": 1, "annotator2": 1, "annotator3": 1},
            "metric2": {"annotator1": 1, "annotator2": 1, "annotator3": 1},
        }
    }
