import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader


class InferencePipeline:
    def __init__(
        self,
        model: nn.Module,
        sequence_length: int,
        device: torch.device,
        logger: logging.Logger,
        batch_size: int,
        prediction_key: str,
        output_dir: Path,
        input_files_list_path: Path,
        processed_files_list_path: Path,
    ):
        self._model = model
        self._sequence_length = sequence_length
        self._device = device
        self._output_dir = output_dir
        self._input_files_list_path = input_files_list_path
        self._processed_files_list_path = processed_files_list_path
        self._batch_size = batch_size
        self._prediction_key = prediction_key
        self._logger = logger
        self._input_file_list = self._get_non_processed_file_path_list(input_files_list_path, processed_files_list_path)

    def _get_file_list(self, input_file_list_path: Path) -> list[Path]:
        with open(input_file_list_path, "r") as f:
            input_file_list = [Path(line.strip()) for line in f]
        return input_file_list

    def _get_non_processed_file_path_list(self, file_list_path: Path, processed_file_list_path: Path) -> list[Path]:
        # load input file list
        input_file_list = self._get_file_list(file_list_path)
        # load processed file list
        if processed_file_list_path.exists():
            processed_file_set = set(self._get_file_list(processed_file_list_path))
        else:
            processed_file_set = set()
        # get non-processed files
        non_processed_file_list = []
        for input_file_path in input_file_list:
            if input_file_path not in processed_file_set:
                non_processed_file_list.append(input_file_path)
            else:
                self._logger.info(f"Skipping already processed file: {input_file_path}")
        return non_processed_file_list

    @staticmethod
    def _write_out_prediction_results(predictions: list[int], output_file: Path, prediction_key: str) -> None:
        with open(output_file, "w") as f:
            for pred in predictions:
                f.write(json.dumps({prediction_key: pred}) + "\n")

    @staticmethod
    def _get_predictions(dataloader: DataLoader, model: torch.nn.Module, device: torch.device) -> list[int]:
        predictions = []

        progress_bar = tqdm.tqdm(
            total=dataloader.batch_size * len(dataloader), desc="Generating scores", unit="samples"
        )
        with torch.inference_mode():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_predictions = model(**batch)
                predictions.extend(batch_predictions.cpu().tolist())
                progress_bar.update(dataloader.batch_size)
        progress_bar.close()
        return predictions

    # def run(self) -> None:
    #     # create output and progress report directory
    #     self._output_dir.mkdir(parents=True, exist_ok=True)
    #     self._processed_files_list_path.parent.mkdir(parents=True, exist_ok=True)

    #     for input_file_path in tqdm.tqdm(self._input_file_list, desc="Processing files"):
    #         self._logger.info(f"Processing file: {input_file_path}")
    #         output_file_path: str = self._output_dir / f"{input_file_path.stem}_annotations.jsonl"

    #         # create dataloader
    #         collate_fn = DataFactory.get_standard_collate_fn(sequence_length=self._sequence_length)
    #         dataloader = DataFactory.get_dataloader(
    #             input_file_path=input_file_path, batch_size=self._batch_size, collate_fn=collate_fn
    #         )

    #         # get predictions
    #         start_time = time.time()
    #         predictions = InferencePipeline._get_predictions(dataloader, self._model, self._device)
    #         end_time = time.time()
    #         self._logger.info(
    #             f"Throuput:  {(end_time - start_time)/(dataloader.batch_size*len(dataloader))*1000:.2f} "
    #             "seconds / 1000 samples for {input_file_path}."
    #         )

    #         # write out results
    #         InferencePipeline._write_out_prediction_results(
    #             predictions, output_file_path, prediction_key=self._prediction_key
    #         )

    #         # update processed files list
    #         with open(self._processed_files_list_path, "a") as f:
    #             f.write(f"{input_file_path}\n")

    #     self._logger.info("Completed successfully.")
