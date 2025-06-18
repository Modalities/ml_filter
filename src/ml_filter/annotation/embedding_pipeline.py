from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from omegaconf import OmegaConf

from ml_filter.annotation.datatrove_jql_annotator import JQLEmbedder, HDF5Writer, stats_adapter


def run_embedding_pipeline(config_file_path: Path):
    """
    Runs the annotation pipeline for scoring text data using a multilingual embedding model
    and regression heads.
    """
    # Define the pipeline steps

    cfg = OmegaConf.load(config_file_path)

    pipeline = [
        JsonlReader(
            data_folder=cfg.embeddings_directory,
            glob_pattern='*.jsonl',
        ),
        JQLEmbedder(
            embedder_model_id="Snowflake/snowflake-arctic-embed-m-v2.0",
            batch_size=1000,
        ),
        HDF5Writer(output_folder=cfg.output_dir + '/embeddings',
                   output_filename="${source_filename}.h5")

    ]
    stage = LocalPipelineExecutor(
        pipeline,
        tasks=cfg.tasks,
        local_tasks=cfg.local_tasks,
        local_rank_offset=cfg.local_rank_offset,
        logging_dir=cfg.output_dir + '/logs',
    )
    stage.run()


if __name__ == '__main__':
    run_embedding_pipeline(config_file_path=Path("/raid/s3/opengptx/jude/repos/ml_filter/ml_filter/configs/annotation/lorem_ipsum_embedding.yaml"))