# 📊 Analysis Code for Prompt-Based Annotations and Interrater Metrics

This module contains scripts for evaluating human and model-generated annotations, measuring interrater agreement, collecting IR metrics, and plotting score distributions.


## Overview of Scripts

| Script | Description |
| :----- | :---------- |
| **`aggregate_scores.py`** | Aggregates individual score files into a single summary CSV. |
| **`collect_ir_metrics.py`** | Collects and processes interrater metrics from model outputs. |
| **`evaluate_prompt_based_annotations.py`** | Evaluates prompt-based human annotations against models or gold standards. |
| **`interrater_reliability.py`** | Computes interrater reliability metrics (e.g., Krippendorff's Alpha, Fleiss' Kappa). |
| **`plot_score_distributions.py`** | Visualizes the distribution of scores across different conditions. |
| **`utils.py`** | Utility functions used across scripts. |



## Expected Input

- **Annotation Data**: CSV files with columns like `prompt_id`, `annotator_id`, `score`.
- **Metric Results**: JSON or CSV files containing evaluation metrics (`precision`, `recall`, `f1`).
- **Config Parameters**: Some scripts require specifying input paths, output directories, or file patterns.

**Example input files:**

```csv
prompt_id,annotator_id,score
1,A,4
1,B,3
2,A,2
2,B,3
```

```json
{
  "precision": 0.85,
  "recall": 0.78,
  "f1": 0.81
}
```


## 🚀 How to Conduct the Analysis

### 1. **Aggregate Annotation Scores**

```bash
python aggregate_scores.py --input_dir path/to/scores/ --output_file aggregated_scores.csv
```
- **Input**: Directory containing per-annotator CSVs.
- **Output**: Single aggregated CSV with mean scores and summary statistics.


### 2. **Evaluate Prompt-Based Annotations**

```bash
python evaluate_prompt_based_annotations.py --annotations_file path/to/annotations.csv --output_dir results/
```
- **Input**: CSV file with prompt-based annotations.
- **Output**: Evaluation results saved in `results/` directory.

---

### 3. **Compute Interrater Reliability**

```bash
python interrater_reliability.py --input_file path/to/annotations.csv --output_file reliability_metrics.json
```
- **Input**: Annotations CSV.
- **Output**: JSON file with interrater reliability scores.

---

### 4. **Collect IR Metrics**

```bash
python collect_ir_metrics.py --input_dir path/to/ir_results/ --output_file collected_ir_metrics.csv
```
- **Input**: Folder containing IR evaluation result files.
- **Output**: Aggregated IR metrics in CSV format.


### 5. **Plot Score Distributions**

```bash
python plot_score_distributions.py --input_file path/to/aggregated_scores.csv --output_dir plots/
```
- **Input**: Aggregated score CSV.
- **Output**: Distribution plots (e.g., `.png` files) saved in `plots/`.


## 📈 Outputs

- **CSV** files: Aggregated scores, collected metrics
- **JSON** files: Interrater reliability statistics
- **Plots**: Score distribution images
- **Summary Reports**: Metrics and evaluation results



