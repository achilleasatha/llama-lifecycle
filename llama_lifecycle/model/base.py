"""Base interface for setting up training and pre-tokenization tasks."""

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from llama_lifecycle.model.tokenizer import Tokenizer

DATA_CACHE_DIR = "./llama_lifecycle/model/artifacts"


def pretokenize():
    enc = Tokenizer()

    data_file = os.path.join(DATA_CACHE_DIR, "tinyshakespeare.txt")

    all_tokens = []
    with open(data_file, "r") as f:
        for line in f:
            text = line.strip()
            tokens = enc.encode(text, bos=True, eos=False)
            all_tokens.extend(tokens)
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens)}")
    with open(data_file.replace(".txt", ".bin"), "wb") as f:
        f.write(all_tokens.tobytes())
    print(f"Saved {data_file.replace('.txt', '.bin')}")
    print("Done.")


class PreTokenizedDataset(IterableDataset):
    """Loads pre-tokenized examples from disk and yields them as PyTorch
    tensors."""

    def __init__(
        self,
        split: str,
        max_seq_len: int,
        export_file_name: str = "pre_tokenized_dataset.bin",
    ):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.export_file_name = export_file_name

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PreTokenizedDataset with seed: {seed}")
        data_file = os.path.join(DATA_CACHE_DIR, self.export_file_name)
        m_all = np.memmap(data_file, dtype=np.uint16, mode="r")

        # split out 10% of the data for validation
        split_ix = int(len(m_all) * 0.9)
        if self.split == "train":
            m = m_all[:split_ix]
        else:
            m = m_all[split_ix:]

        num_batches = len(m) // self.max_seq_len
        num_batches -= 1  # drop the last partial batch
        assert num_batches > 0, "this split is way too small? investigate."

        while True:
            ixs = list(range(num_batches))
            rng.shuffle(ixs)
            for ix in ixs:
                start = ix * self.max_seq_len
                end = start + self.max_seq_len + 1
                # calling .astype will copy the data into a new numpy array, now in RAM
                chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                x = chunk[:-1]
                y = chunk[1:]
                yield x, y


class BaseTask:
    @staticmethod
    def iter_batches(
        split: str,
        batch_size: int,
        max_seq_len: int,
        device: str,
        num_workers: int = 0,
    ):
        ds = PreTokenizedDataset(split=split, max_seq_len=max_seq_len)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["train_tokenizer", "pretokenize"])
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    fun = {
        "pretokenize": pretokenize,
    }
    fun[args.stage]()
