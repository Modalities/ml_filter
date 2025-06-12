from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.writers import JsonlWriter
from omegaconf import OmegaConf

from ml_filter.annotation.datatrove_jql_annotator import JQLHead, stats_adapter, JQLEmbeddingReader


def run_annotation_pipeline(config_file_path: Path):
    """
    Runs the annotation pipeline for scoring text data using a multilingual embedding model
    and regression heads.
    """
    # Define the pipeline steps

    cfg = OmegaConf.load(config_file_path)

    pipeline = [
        JQLEmbeddingReader(
            data_folder=cfg.embeddings_directory, ),
        JQLHead(
            regression_head_checkpoints=None,  # Uses default heads
            batch_size=cfg.batch_size,
            stats_writer=JsonlWriter(
                output_folder=cfg.output_dir + '/annotated_data',
                output_filename="${source_filename}.jsonl",
                adapter=stats_adapter,
                expand_metadata=True,

            ),
        )
    ]
    stage = LocalPipelineExecutor(
        pipeline,
        tasks=cfg.tasks,
        local_tasks=cfg.local_tasks,
        local_rank_offset=cfg.local_rank_offset,
        logging_dir=cfg.output_dir + '/logs',
    )
    stage.run()
