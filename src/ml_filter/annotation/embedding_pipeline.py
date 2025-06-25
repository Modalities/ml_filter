from pathlib import Path

from datatrove.executor import SlurmPipelineExecutor
from omegaconf import OmegaConf

from ml_filter.annotation.datatrove_jql_annotator import JQLEmbedder, HDF5Writer, JQLJsonlReader


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

    pipeline = [
        JQLJsonlReader(
            data_folder=cfg.input_dir,
            csv_hashmap=Path(cfg.csv_hashmap_path),
            glob_pattern=cfg.glob_pattern,
        ),
        JQLEmbedder(
            embedder_model_id=cfg.embedding_model,
            batch_size=cfg.batch_size,
        ),
        HDF5Writer(output_folder=cfg.output_dir + '/embeddings',
                   output_filename="${source_filename}.h5",
                   dataset_name=cfg.hdf5_dataset_name
        )

    ]
    stage = SlurmPipelineExecutor(
        pipeline,
        job_name=cfg.slurm.job_name,
        logging_dir=cfg.output_dir + '/logs',
        tasks=cfg.tasks,
        workers=cfg.slurm.workers,
        time=cfg.slurm.time,
        partition=cfg.slurm.partition,
        qos=cfg.slurm.qos,
        cpus_per_task=cfg.slurm.cpus_per_task,
        # mem_per_cpu=cfg.slurm.mem_per_cpu,
        # gpu_per_task=cfg.slurm.gpu_per_task,
        # account=cfg.slurm.account,
    )
    stage.run()


# Testing
if __name__ == '__main__':
    run_embedding_pipeline(config_file_path=Path("/raid/s3/opengptx/jude/repos/ml_filter/ml_filter/configs/annotation/lorem_ipsum_embedding.yaml"))