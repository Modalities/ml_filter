import click
import click_pathlib
from pathlib import Path
from omegaconf import OmegaConf
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from edu_filter.lms_llama_run import Main

@click.group()
def main() -> None:
    pass

@main.command(name="score_documents")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to a file with the YAML config file.",
)
def entry_point_score_documents(config_file_path: Path):
    processor = Main(config_file_path=config_file_path)
    processor.run()
    

if __name__ == "__main__":
    main()