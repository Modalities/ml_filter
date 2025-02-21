
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

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
        expected = pd.DataFrame.from_dict({
            'prompt': {0: 'prompt1', 1: 'prompt1', 2: 'prompt1', 3: 'prompt2', 4: 'prompt2'},
            'prompt_lang': {0: 'lang1', 1: 'lang1', 2: 'lang1', 3: 'lang2', 4: 'lang2'},
            'model': {0: 'model1', 1: 'model1', 2: 'model1', 3: 'model2', 4: 'model2'},
            'doc_id': {0: 'doc1', 1: 'doc2', 2: 'doc3', 3: 'doc1', 4: 'doc2'},
            'score': {0: 3.3333333333333335, 1: 1.6666666666666667, 2: 2.5, 3: 5.0, 4: 4.0}
        })
        pd.testing.assert_frame_equal(result, expected)

        # Test aggregation: max
        result = get_document_scores([file1, file2], aggregation="max", labels=labels)
        expected = pd.DataFrame.from_dict({
            'prompt': {0: 'prompt1', 1: 'prompt1', 2: 'prompt1', 3: 'prompt2', 4: 'prompt2'},
            'prompt_lang': {0: 'lang1', 1: 'lang1', 2: 'lang1', 3: 'lang2', 4: 'lang2'},
            'model': {0: 'model1', 1: 'model1', 2: 'model1', 3: 'model2', 4: 'model2'},
            'doc_id': {0: 'doc1', 1: 'doc2', 2: 'doc3', 3: 'doc1', 4: 'doc2'},
            'score': {0: 4.0, 1: 2.0, 2: 3.0, 3: 5.0, 4: 4.0}
        })
        pd.testing.assert_frame_equal(result, expected)

        # Test aggregation: min
        result = get_document_scores([file1, file2], aggregation="min", labels=labels)
        expected = pd.DataFrame.from_dict({
            'prompt': {0: 'prompt1', 1: 'prompt1', 2: 'prompt1', 3: 'prompt2', 4: 'prompt2'},
            'prompt_lang': {0: 'lang1', 1: 'lang1', 2: 'lang1', 3: 'lang2', 4: 'lang2'},
            'model': {0: 'model1', 1: 'model1', 2: 'model1', 3: 'model2', 4: 'model2'},
            'doc_id': {0: 'doc1', 1: 'doc2', 2: 'doc3', 3: 'doc1', 4: 'doc2'},
            'score': {0: 3.0, 1: 1.0, 2: 2.0, 3: 5.0, 4: 4.0}
        })
        pd.testing.assert_frame_equal(result, expected)
        
        # Test aggregation: majority
        result = get_document_scores([file1, file2], aggregation="majority", labels=labels)
        expected = pd.DataFrame.from_dict({
            'prompt': {0: 'prompt1', 1: 'prompt1', 2: 'prompt1', 3: 'prompt2', 4: 'prompt2'},
            'prompt_lang': {0: 'lang1', 1: 'lang1', 2: 'lang1', 3: 'lang2', 4: 'lang2'},
            'model': {0: 'model1', 1: 'model1', 2: 'model1', 3: 'model2', 4: 'model2'},
            'doc_id': {0: 'doc1', 1: 'doc2', 2: 'doc3', 3: 'doc1', 4: 'doc2'},
            'score': {0: 3.0, 1: 2.0, 2: 2.5, 3: 5.0, 4: 4.0}
        })
        pd.testing.assert_frame_equal(result, expected)
