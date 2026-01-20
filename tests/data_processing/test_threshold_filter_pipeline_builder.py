from __future__ import annotations

from pathlib import Path

from ml_filter.data_processing.jsonl_filtering.threshold_filter_pipeline import ThresholdFilterPipelineBuilder


def test_builder_from_yaml_parses_paired_mode(tmp_path: Path):
    # minimal builder-style YAML
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
running_on_slurm: false
params:
  text_input_dir: /tmp/text
  scores_input_dir: /tmp/scores
  domains_input_dir: /tmp/domains
  paths_file: /tmp/paths.txt
  glob_pattern: null
  recursive: false
  compression: null

  score_keys: [score_Gemma_Snowflake]
  thresholds_by_score_key:
    score_Gemma_Snowflake: 0.5

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

    b = ThresholdFilterPipelineBuilder.from_yaml(cfg)
    assert b.params.text_input_dir == "/tmp/text"
    assert b.params.scores_input_dir == "/tmp/scores"
    assert b.params.domains_input_dir == "/tmp/domains"
    assert b.params.paths_file == "/tmp/paths.txt"
    assert b.params.text_jsonl_id_key == "id"
    assert b.params.score_jsonl_id_key == "id"
    assert b.params.text_jsonl_text_key == "text"
    assert b.params.domain_jsonl_id_key == "id"
    assert b.params.domain_jsonl_domain_key == "domain"
    assert b.params.accepted_domains == ["wikipedia.org", "stackexchange.com"]

    pipeline = b.build_pipeline()
    assert len(pipeline) == 4
    assert pipeline[0].name == "Paired_Filter_By_Threshold"
    assert pipeline[1].name == "Filter_By_Domain"
    assert pipeline[2].name == "Filter_By_NumWords"
    # Datatrove's JsonlWriter uses a friendly name (currently "🐿 Jsonl"),
    # so just assert it's a writer-ish final step.
    assert "jsonl" in pipeline[3].name.lower()

    assert b.params.output_filename == "${file_relpath}"