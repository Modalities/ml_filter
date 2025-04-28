import json

import pandas as pd

# Import the functions to be tested
from ml_filter.analysis.utils import (  # Replace 'your_module' with the actual module name
    get_common_docs,
    get_document_scores,
    most_frequent_average,
    round_scores,
)


def test_most_frequent_average():
    # Test with a single most frequent value
    assert most_frequent_average([4, 2, 2, 3, 3, 5]) == 2.5
    # Test with no ties
    assert most_frequent_average([1, 1, 2, 2, 2, 3]) == 2.0
    # Test with ties
    assert most_frequent_average([1, 2, 2, 3, 3, 4]) == 2.5
    # Test with a single element
    assert most_frequent_average([7]) == 7.0
    # Test with all elements the same
    assert most_frequent_average([5, 5, 5]) == 5.0


def test_get_document_scores(tmp_path):
    # Create temporary JSONL files for testing
    file1 = tmp_path / "scores__prompt1__lang1__annotator1.jsonl"
    file2 = tmp_path / "scores__prompt2__lang2__annotator2.jsonl"

    # Write JSONL data to files
    data1 = [
        {"document_id": "doc1", "scores": [3, 3, 4]},
        {"document_id": "doc2", "scores": [1, 2, 2]},
        {"document_id": "doc3", "scores": [None, 2, 3]},
    ]
    data2 = [
        {"document_id": "doc1", "scores": [5, 5, 5]},
        {"document_id": "doc2", "scores": [4, 4, 4]},
    ]

    file1.write_text("\n".join(json.dumps(item) for item in data1))
    file2.write_text("\n".join(json.dumps(item) for item in data2))

    labels = [1, 2, 3, 4, 5]

    # Test aggregation: mean
    result = get_document_scores([file1, file2], aggregation="mean", labels=labels)
    expected = pd.DataFrame.from_dict(
        {
            "prompt": {0: "prompt1", 1: "prompt1", 2: "prompt1", 3: "prompt2", 4: "prompt2"},
            "prompt_lang": {0: "lang1", 1: "lang1", 2: "lang1", 3: "lang2", 4: "lang2"},
            "annotator": {0: "annotator1", 1: "annotator1", 2: "annotator1", 3: "annotator2", 4: "annotator2"},
            "doc_id": {0: "doc1", 1: "doc2", 2: "doc3", 3: "doc1", 4: "doc2"},
            "score": {0: 3.3333333333333335, 1: 1.6666666666666667, 2: 2.5, 3: 5.0, 4: 4.0},
            "raw_data_file_path": {0: None, 1: None, 2: None, 3: None, 4: None},
        }
    )
    pd.testing.assert_frame_equal(result, expected)

    # Test aggregation: max
    result = get_document_scores([file1, file2], aggregation="max", labels=labels)
    expected = pd.DataFrame.from_dict(
        {
            "prompt": {0: "prompt1", 1: "prompt1", 2: "prompt1", 3: "prompt2", 4: "prompt2"},
            "prompt_lang": {0: "lang1", 1: "lang1", 2: "lang1", 3: "lang2", 4: "lang2"},
            "annotator": {0: "annotator1", 1: "annotator1", 2: "annotator1", 3: "annotator2", 4: "annotator2"},
            "doc_id": {0: "doc1", 1: "doc2", 2: "doc3", 3: "doc1", 4: "doc2"},
            "score": {0: 4.0, 1: 2.0, 2: 3.0, 3: 5.0, 4: 4.0},
            "raw_data_file_path": {0: None, 1: None, 2: None, 3: None, 4: None},
        }
    )
    pd.testing.assert_frame_equal(result, expected)

    # Test aggregation: min
    result = get_document_scores([file1, file2], aggregation="min", labels=labels)
    expected = pd.DataFrame.from_dict(
        {
            "prompt": {0: "prompt1", 1: "prompt1", 2: "prompt1", 3: "prompt2", 4: "prompt2"},
            "prompt_lang": {0: "lang1", 1: "lang1", 2: "lang1", 3: "lang2", 4: "lang2"},
            "annotator": {0: "annotator1", 1: "annotator1", 2: "annotator1", 3: "annotator2", 4: "annotator2"},
            "doc_id": {0: "doc1", 1: "doc2", 2: "doc3", 3: "doc1", 4: "doc2"},
            "score": {0: 3.0, 1: 1.0, 2: 2.0, 3: 5.0, 4: 4.0},
            "raw_data_file_path": {0: None, 1: None, 2: None, 3: None, 4: None},
        }
    )
    pd.testing.assert_frame_equal(result, expected)

    # Test aggregation: majority
    result = get_document_scores([file1, file2], aggregation="majority", labels=labels)
    expected = pd.DataFrame.from_dict(
        {
            "prompt": {0: "prompt1", 1: "prompt1", 2: "prompt1", 3: "prompt2", 4: "prompt2"},
            "prompt_lang": {0: "lang1", 1: "lang1", 2: "lang1", 3: "lang2", 4: "lang2"},
            "annotator": {0: "annotator1", 1: "annotator1", 2: "annotator1", 3: "annotator2", 4: "annotator2"},
            "doc_id": {0: "doc1", 1: "doc2", 2: "doc3", 3: "doc1", 4: "doc2"},
            "score": {0: 3.0, 1: 2.0, 2: 2.5, 3: 5.0, 4: 4.0},
            "raw_data_file_path": {0: None, 1: None, 2: None, 3: None, 4: None},
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_round_scores():
    assert round_scores(1.5) == 2
    assert round_scores("invalid") == "invalid"


def test_get_common_docs():
    data = {
        "annotator": ["annotator_0", "annotator_0", "annotator_1", "annotator_1"],
        "doc_id": [1, 2, 1, 2],
        "prompt": ["p1", "p2", "p1", "p2"],
        "score": [1, 2, 1, 2],
    }
    df = pd.DataFrame(data)
    common_docs_df = get_common_docs(df, "annotator_0", "annotator_1")
    assert isinstance(common_docs_df, pd.DataFrame)
    assert "rounded_score_0" in common_docs_df.columns
    assert "rounded_score_1" in common_docs_df.columns
