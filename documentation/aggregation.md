This documentation gives an overview of the typical steps that have to be done in order to aggregate the annotations of our pipeline for further classifier training.

1. Log in to MN5 and load the required environment:
```bash
ml impi intel load mkl hdf5 python/3.11.5-gcc
source /home/frau/frau435699/ehpc17/richard/repositories/marenostrum5-tools/mn5_ml_filter_deployment/venvs/ml_filter_build_974c508ce77f9e9d92996a2e0ff80190b37f9654/bin/activate 
# load your environment instead
```
 The environment is built and deployed using [marenostrum5-tools](https://github.com/EuroLingua-GPT/marenostrum5-tools).

 2. The results of our experiments should be collected in the following folder:
```bash
/gpfs/projects/ehpc17/results/prompt_based_annotations
```

3. We have multiple annotations per document to account for a certain randomness in our decoding strategy. These scores have to be aggregated to a single score. This can be done with [ml_filter](https://github.com/EuroLingua-GPT/ml_filter). To aggregate the scores in all jsonl-files in a certain directory (e.g. for all of the 37 languages), you can run the following commands:
```bash
start_dir="/gpfs/projects/ehpc17/results/prompt_based_annotations/educational_content/Llama-3.3-70B-Instruct"
target_dir="${start_dir}_aggregated"
cd /path/to/ml_filter
find $start_dir -type f -name "*.jsonl" | while read -r file; do
    parent_dir=$(dirname "$file")
    python3.11 src/ml_filter/__main__.py aggregate_scores $parent_dir $target_dir --aggregation majority --labels 0,1,2,3,4,5 --raw_data_lookup_dir /gpfs/scratch/ehpc17/dqa/data/fineweb_2_500k_both_deduplicated
done
```

4. The results of step 3 can be transferred to a machine with internet access and from the uploaded to huggingface.
