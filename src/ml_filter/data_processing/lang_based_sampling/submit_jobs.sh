#!/bin/bash
set -euo pipefail

mkdir -p logs

echo "Submitting jobs :"
for lang in $(python3 -c "import yaml, os; cfg=yaml.safe_load(open(os.environ['PIPELINE_CONFIG'])); print(' '.join(cfg['sampling']['language_distribution'].keys()))"); do
    echo "Submitting job for $lang"
    sbatch run_language_pipeline.slurm "$lang"
done

echo "✅ All jobs submitted."
