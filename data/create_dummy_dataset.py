import json
import os
import random
from pathlib import Path

working_dir = Path(os.path.dirname(__file__))

source_dataset_path = working_dir / "lorem_ipsum.jsonl"
destination_path = working_dir / "lorem_ipsum_multiscore.jsonl"

scoring_keys = [
    "edu_en",
    "edu_de",
    "toxicity_en",
    "toxicity_de",
]

with open(source_dataset_path, "r") as f:
    data = [json.loads(line) for line in f]

for record in data:
    record.pop("score", None)
    scores = [random.randint(0, 5 if k.startswith('e') else 1) for k in scoring_keys]
    # record["labels"] = scores
    record["scores"] = {k: v for k, v in zip(scoring_keys, scores)}
    # for k, v in zip(scoring_keys, scores):
    #     record[k] = v

with open(destination_path, "w") as f:
    for record in data:
        f.write(json.dumps(record) + "\n")
