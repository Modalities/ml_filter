import shutil
from pathlib import Path
from typing import Dict, List

import yaml

from ml_filter.compare_experiments import CompareConfig, StatisticConfig, compare_experiments
from ml_filter.data_processing.document import Annotation, MetaInformation
from ml_filter.data_processing.report_statistics import ThroughputStatistics


def test_compare_experiments(tmpdir: Path):
    # Wirte example result file containing the fields scores, document_id, and language
    meta_info = MetaInformation(
        prompt_name="", prompt_lang="", model="", raw_data_file_path="", out_tokens_per_second=0
    )
    annotation_1 = Annotation(scores=[5], document_id="0", meta_information=meta_info)
    annotation_2 = Annotation(scores=[3], document_id="1", meta_information=meta_info)

    test_annotations_path = tmpdir / "test__annotations_results.jsonl"
    with open(test_annotations_path, "w") as f:
        f.write(annotation_1.model_dump_json() + "\n")
        f.write(annotation_2.model_dump_json() + "\n")

    exp_config_file_path = tmpdir / "exp_config.yaml"
    exp_config = dict(settings=dict(model_name="", num_gpus=1), tokenizer=dict(add_generation_prompt=True))
    with open(exp_config_file_path, "w") as f:
        yaml.dump(exp_config, f, default_flow_style=False)

    gold_annotations_path = tmpdir / "gold_annotations.jsonl"
    shutil.copy(test_annotations_path, gold_annotations_path)

    config_file_path = tmpdir / "config.yaml"

    config = CompareConfig(
        experiment_dir_paths=[Path(tmpdir)],
        experiment_config_file_name=Path(exp_config_file_path).name,
        output_format=[StatisticConfig(sort_by_metric="mean_out_tokens_per_second", ascending=False)],
    )
    with open(config_file_path, "w") as f:
        yaml.dump(_dict_with_pathlib_path_conversion(config.model_dump()), f, default_flow_style=False)
    throughput_stats = ThroughputStatistics(
        mean_out_tokens_per_second=0.0,
        num_documents_written=0,
        elapsed_time_s=0.0,
        documents_per_second=0.0,
        model_name="",
        queue_size=0,
        num_processes=0,
        max_new_tokens=0,
    )

    with open(tmpdir / "throughput.json", "w") as f:
        f.write(throughput_stats.model_dump_json())

    df = compare_experiments(Path(config_file_path))
    assert df.iloc[0]["mean_out_tokens_per_second"] == 0.0


def _dict_with_pathlib_path_conversion(inp: Dict | List | Path) -> Dict | List | str:
    if isinstance(inp, Path):
        return str(inp)
    if isinstance(inp, dict):
        return {k: _dict_with_pathlib_path_conversion(v) for k, v in inp.items()}
    if isinstance(inp, list):
        return [_dict_with_pathlib_path_conversion(v) for v in inp]
    return inp
