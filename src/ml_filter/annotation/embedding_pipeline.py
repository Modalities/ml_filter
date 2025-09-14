from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from omegaconf import OmegaConf

from ml_filter.annotation.datatrove_jql_annotator import HDF5Writer, JQLEmbedder, JQLJsonlReader


def run_embedding_pipeline(config_file_path: Path):
    """
    Runs the annotation pipeline for scoring text data using a multilingual embedding model
    and regression heads.
    """
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_file_path}")

    try:
        cfg = OmegaConf.load(config_file_path)
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_file_path}: {e}")

    embedding_dir = getattr(cfg, "embedding_dir")
    embedding_output_path = str(Path(cfg.output_dir) / embedding_dir)
    pipeline = [
        JQLJsonlReader(
            data_folder=cfg.input_dir,
            csv_hashmap=Path(cfg.csv_hashmap_path),
            glob_pattern=cfg.glob_pattern,
        ),
        JQLEmbedder(
            embedder_model_id=cfg.embedding_model,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            padding=cfg.padding,
            truncation=cfg.truncation,
            stats_writer=HDF5Writer(
                output_folder=embedding_output_path,
                output_filename="${source_filename}.h5",
                dataset_name=cfg.hdf5_dataset_name,
                batch_size=cfg.writer_batch_size,
            ),
        ),
    ]
    stage = LocalPipelineExecutor(
        pipeline,
        tasks=cfg.tasks,
        local_tasks=cfg.local_tasks,
        local_rank_offset=cfg.local_rank_offset,
        workers=cfg.workers,
        logging_dir=cfg.output_dir + "/logs",
    )

    stage.run()


# Testing
if __name__ == "__main__":
    run_embedding_pipeline(
        config_file_path=Path(
            "/raid/s3/opengptx/jude/repos/ml_filter/ml_filter/configs/annotation/lorem_ipsum_embedding.yaml"
        )
    )
