from functools import partial
from pathlib import Path
from typing import Callable

import torch
from modalities.dataloader.dataset import PackedMemMapDatasetBase
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class DataFactory:
    def get_dataloader(
        input_file_path: Path, batch_size: int, collate_fn: Callable, sample_key: str = "input_ids"
    ) -> DataLoader:
        dataset = PackedMemMapDatasetBase(input_file_path, sample_key=sample_key, load_index=True)
        dataloader: DataLoader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        return dataloader

    def get_standard_collate_fn(
        sequence_length: int,
        padding_value: int = 0,
        sample_key: str = "input_ids",
        attention_mask_key: str = "attention_mask",
    ) -> Callable:
        def collate_fn(
            batch: list[dict[str, list[int]]], sample_key: str, attention_mask_key: str
        ) -> dict[str, torch.Tensor]:
            input_ids = [torch.tensor(sample[sample_key], dtype=torch.long) for sample in batch]
            attention_mask = [torch.ones(len(sample[sample_key]), dtype=torch.long) for sample in batch]

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
            input_ids = input_ids[:, :sequence_length]
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=padding_value)
            attention_mask = attention_mask[:, :sequence_length]

            return {sample_key: input_ids, attention_mask_key: attention_mask}

        return partial(collate_fn, sample_key=sample_key, attention_mask_key=attention_mask_key)
