import json
import os
import random
from pathlib import Path
from typing import List


def create_dummy_dataset(raw_dataset_path: Path, multi_score_dataset_path: Path, scoring_keys: List[str]):
    with open(source_dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    for record in data:
        record.pop("score")
        scores = {k: random.randint(0, 5 if (k in ["edu_en", "edu_de"]) else 1) for k in scoring_keys}
        record["scores"] = scores

    with open(destination_path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    working_dir = Path(os.path.dirname(__file__))

    source_dataset_path = working_dir / "lorem_ipsum.jsonl"
    destination_path = working_dir / "lorem_ipsum_multiscore.jsonl"

    scoring_keys = [
        "edu_en",
        "edu_de",
        "toxicity_en",
        "toxicity_de",
    ]
    create_dummy_dataset(source_dataset_path, destination_path, scoring_keys)
