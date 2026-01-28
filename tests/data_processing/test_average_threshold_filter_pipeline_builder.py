from __future__ import annotations

from pathlib import Path

from ml_filter.data_pipelines.filtering.average_threshold_filter_pipeline import (
    AverageThresholdFilterPipelineBuilder,
)


def test_builder_from_yaml_parses_average_thresholds(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
running_on_slurm: false
params:
  text_input_dir: /tmp/text
  scores_input_dir: /tmp/scores
  domains_input_dir: /tmp/domains
  glob_pattern: null
  recursive: false
  compression: null

  score_keys: [score_a, score_b]
  average_threshold: 1.0
  average_thresholds_by_folder:
    Deu_Latn: 1.5

  min_num_words: 10
  num_words_column: text

  text_jsonl_id_key: id
  score_jsonl_id_key: id
  text_jsonl_text_key: text
  domain_jsonl_id_key: id
  domain_jsonl_domain_key: domain
  accepted_domains: ["wikipedia.org", "stackexchange.com"]

  output_dir: /tmp/out
  output_filename: "${file_relpath}"
local_settings:
  tasks: 1
  local_tasks: 1
  local_rank_offset: 0
  workers: -1
  logging_dir: null
""",
        encoding="utf-8",
    )

    b = AverageThresholdFilterPipelineBuilder.from_yaml(cfg)
    assert b.params.text_input_dir == "/tmp/text"
    assert b.params.scores_input_dir == "/tmp/scores"
    assert b.params.domains_input_dir == "/tmp/domains"
    assert b.params.text_jsonl_id_key == "id"
    assert b.params.score_jsonl_id_key == "id"
    assert b.params.text_jsonl_text_key == "text"
    assert b.params.domain_jsonl_id_key == "id"
    assert b.params.domain_jsonl_domain_key == "domain"
    assert b.params.accepted_domains == ["wikipedia.org", "stackexchange.com"]
    assert b.params.average_threshold == 1.0
    assert b.params.average_thresholds_by_folder == {"Deu_Latn": 1.5}

    pipeline = b.build_pipeline()
    assert len(pipeline) == 3
    assert pipeline[0].name == "Paired_Filter_By_Average_Threshold"
    assert pipeline[1].name == "Filter_By_NumWords"
    assert "jsonl" in pipeline[2].name.lower()

    assert b.params.output_filename == "${file_relpath}"
