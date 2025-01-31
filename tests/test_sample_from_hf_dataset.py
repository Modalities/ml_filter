from unittest.mock import patch, MagicMock, ANY

from ml_filter.sample_from_hf_dataset import upload_file_to_hf, save_data_to_file, sample_from_hf_dataset


@patch("ml_filter.sample_from_hf_dataset.load_dataset")
@patch("ml_filter.sample_from_hf_dataset.save_data_to_file")
def test_sample_from_hf_dataset(mock_save_data_to_file, mock_load_dataset):
    # Arrange
    dataset_name = "test_dataset"
    dataset_split = "train"
    output_file_path = "output.jsonl"
    column_name = "label"
    column_values = ["value1", "value2"]
    num_docs_per_value = 5
    seed = 42

    mock_dataset = {
        dataset_split: MagicMock()
    }
    mock_load_dataset.return_value = mock_dataset

    mock_filtered_data = MagicMock()
    mock_filtered_data.shuffle.return_value.select.return_value = [MagicMock() for _ in range(num_docs_per_value)]
    
    def filter_side_effect(func):
        return mock_filtered_data

    mock_dataset[dataset_split].filter.side_effect = filter_side_effect

    # Act
    sample_from_hf_dataset(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        output_file_path=output_file_path,
        column_name=column_name,
        column_values=column_values,
        num_docs_per_value=num_docs_per_value,
        seed=seed
    )

    # Assert
    mock_load_dataset.assert_called_once_with(dataset_name)
    assert mock_dataset[dataset_split].filter.call_count == len(column_values)
    assert mock_filtered_data.shuffle.return_value.select.call_count == len(column_values)
    mock_save_data_to_file.assert_called_once_with(
        output_file_path=output_file_path,
        data=ANY,
        encoding="utf-8",
        ensure_ascii=False
    )


def test_save_data_to_file(tmp_path):
    # Arrange
    output_file_path = tmp_path / "sample.jsonl"
    data = [{"key": "value"}]
    
    # Act
    save_data_to_file(output_file_path, data)
    
    # Assert
    with open(output_file_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert content == '{"key": "value"}\n'


@patch("ml_filter.sample_from_hf_dataset.HfApi")
def test_upload_file_to_hf(mock_hf_api):
    # Arrange
    file_path = "test_file.txt"
    hf_repo_path = "path/in/repo"
    hf_repo_id = "repo_id"
    repo_type = "dataset"
    hf_token = "fake_token"
    
    mock_api_instance = MagicMock()
    mock_hf_api.return_value = mock_api_instance
    
    # Act
    upload_file_to_hf(file_path, hf_repo_path, hf_repo_id, repo_type, hf_token)
    
    # Assert
    mock_api_instance.upload_file.assert_called_once_with(
        path_or_fileobj=file_path,
        path_in_repo=hf_repo_path,
        repo_id=hf_repo_id,
        repo_type=repo_type,
        token=hf_token,
    )