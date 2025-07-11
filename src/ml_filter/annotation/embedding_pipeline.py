from pathlib import Path

from datatrove.executor import SlurmPipelineExecutor
from omegaconf import OmegaConf

from ml_filter.annotation.datatrove_jql_annotator import JQLEmbedder, HDF5Writer, JQLJsonlReader
from datatrove.pipeline.writers import JsonlWriter
from ml_filter.annotation.datatrove_jql_annotator import stats_adapter


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

    if cfg.datatrove.tasks <= 0:
        raise ValueError("Number of tasks must be > 0")

    pipeline = [
        JQLJsonlReader(
            data_folder=cfg.input_dir,
            csv_hashmap=Path(cfg.csv_hashmap_path),
            glob_pattern=cfg.glob_pattern,
        ),
        JQLEmbedder(
            embedder_model_id=cfg.embedding_model,
            batch_size=cfg.batch_size,
            stats_writer=HDF5Writer(
                output_folder=cfg.output_dir + "/embeddings",
                output_filename="${source_filename}.h5",
                dataset_name=cfg.hdf5_dataset_name,
                batch_size=cfg.batch_size,
            ),
        ),
    ]
    stage = SlurmPipelineExecutor(
        pipeline=pipeline,
        job_name=cfg.slurm.job_name,
        logging_dir=cfg.output_dir + "/logs",
        tasks=cfg.datatrove.tasks,
        workers=cfg.datatrove.workers,
        cpus_per_task=cfg.slurm.cpus_per_task,
        mem_per_cpu_gb=cfg.slurm.mem_per_cpu_gb,
        time=cfg.slurm.time,
        partition=cfg.slurm.partition,
        requeue=False,
        venv_path=cfg.slurm.venv_path,
        qos=cfg.slurm.qos,
        sbatch_args={
            "account": cfg.slurm.account,
            "exclusive": "",
            "nodes": cfg.slurm.nodes,
            "ntasks": cfg.slurm.ntasks,
            "gres": cfg.slurm.gres,
        },
    )
    stage.run()


# Testing
if __name__ == "__main__":
    run_embedding_pipeline(
        config_file_path=Path(
            "/data/cat/ws/alju972f-annotation_at_scale/ml_filter/configs/annotation/lorem_ipsum_embedding.yaml"
        )
    )
