from pathlib import Path

from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.writers import JsonlWriter
from omegaconf import OmegaConf

from ml_filter.annotation.datatrove_jql_annotator import JQLHead, stats_adapter, JQLEmbeddingReader


def run_annotation_pipeline(config_file_path: Path):
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
    
    if cfg.slurm.tasks <= 0:
        raise ValueError("Number of tasks must be > 0")

    pipeline = [
        JQLEmbeddingReader(data_folder=cfg.embeddings_directory),
        JQLHead(
            regression_head_checkpoints=cfg.regression_head_checkpoints,
            batch_size=cfg.batch_size,
            stats_writer=JsonlWriter(
                output_folder=cfg.output_dir + '/annotated_data',
                output_filename="${source_filename}.jsonl",
                adapter=stats_adapter,
                expand_metadata=True,

            ),
        )
    ]
    stage = SlurmPipelineExecutor(
        pipeline=pipeline,
        job_name=cfg.slurm.job_name,
        logging_dir=cfg.output_dir + '/logs',
        tasks=cfg.slurm.tasks,
        workers=cfg.slurm.workers,
        cpus_per_task=cfg.slurm.cpus_per_task,
        time=cfg.slurm.time,
        partition=cfg.slurm.partition,
        venv_path=cfg.slurm.venv_path,
        qos=cfg.slurm.qos,
        requeue=False,
        sbatch_args={"account": cfg.slurm.account, "qos": cfg.slurm.qos, "exclusive": "", "nodes": cfg.slurm.nodes, "ntasks-per-node": cfg.slurm.ntasks, "gres": cfg.slurm.gres},
    )
    stage.run()
