import os
import sqlite3
from typing import Union

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def fetch_dataset(
    dataset_name: str | os.PathLike = "OpenAssistant/oasst1",
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    # TODO separate script to download and save in an efficient format (parquet / avro) in S3 then import from there
    datasets = load_dataset(dataset_name)
    return datasets


def ingest_data(
    dataset_name: str | os.PathLike = "OpenAssistant/oasst1",
    db_name: str | os.PathLike = "./oasst1.db",
):

    datasets = fetch_dataset(dataset_name=dataset_name)

    # Connect to SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create table for combined data
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS dataset (
                        id INTEGER PRIMARY KEY,
                        message_id TEXT,
                        user_id TEXT,
                        created_date TEXT,
                        text TEXT,
                        role TEXT,
                        lang TEXT,
                        deleted INTEGER,
                        synthetic INTEGER,
                        message_tree_id TEXT,
                        tree_state TEXT,
                        train_val_flag INTEGER
                    )"""
    )

    for dataset in datasets:
        if dataset == "train":
            train_val_flag = 1
        elif dataset == "validation":
            train_val_flag = 0
        else:
            with ValueError(f"Unknown dataset: {dataset}") as e:
                print(e.errors())

        for item in datasets[dataset]:
            cursor.execute(
                "INSERT INTO dataset (message_id, user_id, created_date, text, role, lang, deleted, synthetic, "
                "message_tree_id, tree_state, train_val_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item["message_id"],
                    item["user_id"],
                    item["created_date"],
                    item["text"],
                    item["role"],
                    item["lang"],
                    item["deleted"],
                    item["synthetic"],
                    item["message_tree_id"],
                    item["tree_state"],
                    train_val_flag,
                ),
            )

    conn.commit()
    conn.close()

    print("Data successfully written to SQLite database.")


if __name__ == "__main__":
    ingest_data()
