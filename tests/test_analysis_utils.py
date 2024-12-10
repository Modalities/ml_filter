
import pytest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

# Import the functions to be tested
from ml_filter.analysis.utils import most_frequent_average, get_document_scores  # Replace 'your_module' with the actual module name


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
    
    
def test_get_document_scores():
    # Create temporary JSONL files for testing
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir)
        file1 = path / "scores_prompt1_lang1_model1.jsonl"
        file2 = path / "scores_prompt2_lang2_model2.jsonl"

        # Write JSONL data to files
        data1 = [
            {"document_id": "doc1", "scores": [3, 3, 4]},
            {"document_id": "doc2", "scores": [1, 2, 2]},
            {"document_id": "doc3", "scores": [float("-inf"), 2, 3]},  # Invalid entry
        ]
        data2 = [
            {"document_id": "doc1", "scores": [5, 5, 5]},
            {"document_id": "doc2", "scores": [4, 4, 4]},
        ]

        file1.write_text("\n".join(json.dumps(item) for item in data1))
        file2.write_text("\n".join(json.dumps(item) for item in data2))

        # Test aggregation: mean
        result = get_document_scores([file1, file2], aggregation="mean")
        expected = {
            "prompt1": {
                "doc1": {"lang1_model1": 3.33},  # mean([3, 3, 4])
                "doc2": {"lang1_model1": 1.67},  # mean([1, 2, 2])
            },
            "prompt2": {
                "doc1": {"lang2_model2": 5.0},   # mean([5, 5, 5])
                "doc2": {"lang2_model2": 4.0},   # mean([4, 4, 4])
            }
        }
        # Round results to two decimal places for testing
        for prompt, docs in result.items():
            for doc_id, versions in docs.items():
                for version, score in versions.items():
                    result[prompt][doc_id][version] = round(score, 2)
        assert result == expected

        # Test aggregation: max
        result = get_document_scores([file1, file2], aggregation="max")
        expected = {
            "prompt1": {
                "doc1": {"lang1_model1": 4},  # max([3, 3, 4])
                "doc2": {"lang1_model1": 2},  # max([1, 2, 2])
            },
            "prompt2": {
                "doc1": {"lang2_model2": 5},  # max([5, 5, 5])
                "doc2": {"lang2_model2": 4},  # max([4, 4, 4])
            }
        }
        assert result == expected

        # Test aggregation: majority
        result = get_document_scores([file1, file2], aggregation="majority")
        expected = {
            "prompt1": {
                "doc1": {"lang1_model1": 3},  # most_frequent_average([3, 3, 4])
                "doc2": {"lang1_model1": 2},  # most_frequent_average([1, 2, 2])
            },
            "prompt2": {
                "doc1": {"lang2_model2": 5},  # most_frequent_average([5, 5, 5])
                "doc2": {"lang2_model2": 4},  # most_frequent_average([4, 4, 4])
            }
        }
        assert result == expected
        
        # Test aggregation: None
        result = get_document_scores([file1, file2], aggregation=None)
        expected = {
            'prompt1': {
                'doc1': {'lang1_model1_0': 3, 'lang1_model1_1': 3, 'lang1_model1_2': 4},
                'doc2': {'lang1_model1_0': 1, 'lang1_model1_1': 2, 'lang1_model1_2': 2}
            },
            'prompt2': {
                'doc1': {'lang2_model2_0': 5, 'lang2_model2_1': 5, 'lang2_model2_2': 5},
                'doc2': {'lang2_model2_0': 4, 'lang2_model2_1': 4, 'lang2_model2_2': 4}
            }
        }
        assert result == expected
