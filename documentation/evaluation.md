This documentation gives an overview of the typical steps that have to be done in order to evaluate prompt based annotations against the ground truth and add the results to a latex file.


1. First put the annotations of different LLMs on the test data in a common folder /path/to/annotations .
This folder should contain all LLMs for which you want to report the evaluation metrics together in one table in our overleaf document:

```script
python3.11 src/ml_filter/__main__.py evaluate_prompt_based_annotations --input_directory /path/to/annotations --output_directory /path/to/annotations/comparison --gt_data /raid/s3/opengptx/user/richard-rutmann/data/eurolingua/prompt_based_annotations/test_data/gt/annotations__educational_content__en__gt.jsonl --aggregation majority --labels 0,1,2,3,4,5
```

Currently we use this path on DGX2 to collect and evaluate annotations:
/raid/s3/opengptx/user/richard-rutmann/data/eurolingua/experiments/multilinguality/experiments


2. Next, we will extract the metrics of interest from the files produced in step 1 and write them into a latex-file: 

```script
python3.11 src/ml_filter/__main__.py collect_ir_metrics --input_directory /path/to/annotations/comparison --output_directory /path/to/annotations/comparison/ir_summary --min_metrics Invalid,MSE,MAE
```

The content of 
```bash
/path/to/annotations/comparison/ir_summary/ir_summary_gt.tex
```

can then be copied partly or fully to the overleaf document.

In addition, you can find confusion matrices and histograms for each annotator and language in 
```bash
/path/to/annotations/comparison
```

as well as aggregated across languages in 
```bash
/path/to/annotations/comparison/ir_summary
```