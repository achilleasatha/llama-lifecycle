import sqlite3
import tempfile
from unittest.mock import patch

import pytest

from llama_lifecycle.db.ingestion_pipeline import ingest_data


@pytest.fixture
def temp_db():
    temp_db_file = tempfile.NamedTemporaryFile()
    conn = sqlite3.connect(temp_db_file.name)
    cursor = conn.cursor()
    yield conn, cursor, temp_db_file


@pytest.fixture
def sample_dataset():
    # Sample dataset
    return {
        "train": [
            {
                "message_id": "1",
                "user_id": "user1",
                "created_date": "2022-01-01",
                "text": "Sample message 1",
                "role": "user",
                "lang": "en",
                "deleted": 0,
                "synthetic": 0,
                "message_tree_id": "tree1",
                "tree_state": "active",
            }
        ],
        "validation": [
            {
                "message_id": "2",
                "user_id": "user2",
                "created_date": "2022-01-02",
                "text": "Sample message 2",
                "role": "user",
                "lang": "en",
                "deleted": 0,
                "synthetic": 0,
                "message_tree_id": "tree2",
                "tree_state": "active",
            }
        ],
    }


@patch("llama_lifecycle.db.ingestion_pipeline.fetch_dataset")
def test_ingest_data(mock_fetch_dataset, temp_db, sample_dataset):
    mock_fetch_dataset.return_value = sample_dataset
    conn, cursor, temp_db_file = temp_db
    ingest_data(db_name=temp_db_file.name)
    cursor.execute("SELECT COUNT(*) FROM dataset")
    row_count = cursor.fetchone()[0]
    assert row_count == 2
    cursor.close()
    conn.close()
    temp_db_file.close()
