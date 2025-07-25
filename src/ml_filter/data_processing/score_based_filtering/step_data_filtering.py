import dataclasses
import logging
from pathlib import Path
from typing import Callable

import numpy as np
from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from numpy.typing import NDArray

from ml_filter.data_processing.score_based_filtering.step_score_parsing import ScoresParser

try:
    from modalities.dataloader.filter_packed_data import filter_dataset
except ImportError:
    logging.error("The filtering pipeline requires the 'modalities' package to be installed.")
    exit(1)


class DataFiltering(PipelineStep):
    """
    A class to filter datasets based on scores and specified thresholds.
    This class is designed to be used within a datatrove pipeline.
    For a given list of score dictionaries, it filters the corresponding tokenized dataset files
    based on the provided thresholds for each score.
    The resulting filtered datasets are saved in the specified output folder.
    Args:
        output_folder (Path): The folder where the filtered datasets will be saved.
        thresholds (dict[str, float]): A dictionary where keys are score names and values are the
            thresholds to filter samples.
        tokenized_data_path (Path): The path for the tokenized data files.
    Raises:
        AssertionError: If the output folder is not a directory or if no thresholds are provided.
    """

    name = "DataFiltering"
    type = "Filter"
    _requires_dependencies = []

    def __init__(self, output_folder: Path, thresholds: dict[str, float], tokenized_data_path: Path = Path("")):
        super().__init__()
        self._output_folder = output_folder
        assert self._output_folder.is_dir(), f"Output folder {self._output_folder} must be a directory."
        self._thresholds = thresholds
        assert len(self._thresholds) > 0, "At least one threshold must be provided."
        self._tokenized_data_path = tokenized_data_path

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for document in data:
            with self.track_time():
                self._filter_document(document)
            yield document

    def _filter_document(self, document: Document):
        """
        Filters a single, tokenized dataset based on the scores contained in the document.
        Args:
            document (Document): The document containing scores and the path to the tokenized data file.
        Raises:
            ValueError: If the document does not contain the required keys or if the tokenized file path is invalid.
        """
        document: dict[str, list[dict[str, float]] | str] = dataclasses.asdict(document)
        scores: list[dict[str, float]] = document["metadata"][ScoresParser.SCORE_ENTRIES_KEY]
        tokenized_file = Path(document["metadata"][ScoresParser.TOKENIZED_FILE_KEY])
        output_path = self._prepare_output_path(tokenized_file)
        filter_func = make_filter_func(scores, self._thresholds)
        filter_dataset(src_path=tokenized_file, dst_path=output_path, filter_func=filter_func, sample_key="input_ids")

    def _prepare_output_path(self, tokenized_file: Path) -> Path:
        tokenized_file_rel = tokenized_file.relative_to(self._tokenized_data_path)
        output_path = self._output_folder / tokenized_file_rel.with_suffix(".filtered.pbin")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path


def make_filter_func(
    scores: list[dict[str, float]], thresholds: dict[str, float]
) -> Callable[[tuple[int, dict[str, NDArray[np.int_]]]], bool]:
    """
    Creates a filter function that checks if the scores of each sample meet the specified thresholds.
    Args:
        scores (list[dict[str, float]]): A list of dictionaries containing scores for each sample.
        thresholds (dict[str, float]): A dictionary where keys are score names and values are the thresholds to
            filter samples.
    Returns:
        Callable[[tuple[int, dict[str, NDArray[np.int_]]]], bool]: A function that takes an item (index and
            sample) and returns True if the sample meets the thresholds, otherwise False.
    """

    def filter_func(item: tuple[int, dict[str, NDArray[np.int_]]]) -> bool:
        idx, _ = item
        score_entry = scores[idx]
        for score_key, threshold in thresholds.items():
            if score_entry[score_key] < threshold:
                return False
        return True

    return filter_func
