import os
import tempfile
from unittest.mock import patch

import pytest

from llama_lifecycle.train.preprocessing_pipeline import preprocess_dataset


@pytest.fixture
def mock_train_dataset():
    return {
        "message_tree_id": ["1", "1", "2", "2", "3"],
        "text": [
            "Root message 1",
            "Child message 1",
            "Root message 2",
            "Child message 2",
            "Root message 3",
        ],
    }


@patch("datasets.load_dataset")
def test_process_and_save_dataset(mock_load_dataset, mock_train_dataset):
    # Mock the load_dataset function to return the mock dataset
    mock_load_dataset.return_value = mock_train_dataset

    # Set up temporary directory
    with tempfile.TemporaryDirectory() as tmp_output_dir:
        # Call the function to preprocess and save the dataset
        preprocess_dataset(output_dir=tmp_output_dir)

        # Check if the file is saved to the correct location
        assert os.path.exists(tmp_output_dir)

        # List of expected files
        expected_files = [
            "state.json",
            "data-00000-of-00001.arrow",
            "dataset_info.json",
        ]

        # Check if all expected files exist in the directory
        for file_name in expected_files:
            assert os.path.exists(
                os.path.join(tmp_output_dir, file_name)
            ), f"File {file_name} not found in {tmp_output_dir}"
