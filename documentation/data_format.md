# Definition of the data formats

## Raw Common Crawl data format

The common crawl dumps come with the following folder structure:

```
├── 2014-41
│   ├── bg
│   │   └── 00055.jsonl
│   ├── ca
│   │   ├── 00000.jsonl
│   │   ├── 00035.jsonl
│   │   └── 00040.jsonl
│   └── cs
│       ├── 00001.jsonl
│       └── 00009.jsonl
├── 2015-02

[...]
```

A dump is defined by the year and an incremental number (format: `yyyy-nn`). Each dump contains a folder for each language. Each language folder contains a number of JSONL files. Each jsonl JSONL contains a number of JSON-formatted documents.

A single document in the JSONL file folows the format below. Besides the raw utf-8 encoded text, it has a unique identifier, metadata, and a number of features extracted from the text.
```
{
   "text":"Oscar Isaac im InterviewAls Oscar Isaac erfuhr, dass [...]",
   "id":"<urn:uuid:bf5b8731-477e-45de-85ce-cf88351c6929>",
   "metadata":{
      "url":"http://www.interview.de/oscar-isaac-im-interview/",
      "date":"2014-09-22T22:15:10Z",
      "language":"de",
      "language_score":0.9818282723426819,
      "corpus":"2014-41",
      "file_path":"/data/octopus/p_gptx/raw_data/cc/2014-41/29799.txt.gz",
      "num_bad_words_in_url":0.0,
      "bad_word_rate":0.0,
      "num_lines":1,
      "num_nonempty_lines":1,
      "num_paras":1,
      "avg_line_len":1161.0,
      "avg_para_len":1161.0,
      "avg_words_per_line":183.0,
      "num_short_lines":0,
      "has_short_header":0,
      "has_short_footer":0,
      "num_itemize_lines":0,
      "num_enumerate_lines":0,
      "pct_itemize_lines":0.0,
      "pct_enumerate_lines":0.0,
      "num_repeated_chars3":0,
      "num_repeated_words3":0,
      "num_urls":0,
      "num_emails":0,
      "num_words":184,
      "avg_word_len":5.11413,
      "ttr_words_lower":0.70652,
      "pct_space":0.0,
      "pct_lower":0.57065,
      "pct_upper":0.0,
      "pct_title":0.42391,
      "pct_mixed":0.00543,
      "pct_puncts":0.13146,
      "pct_words":0.86385,
      "pct_numbers":0.0,
      "pct_others":0.00469,
      "pct_short_lines":0.0,
      "pct_stopwords":0.41315,
      "line_punct_ratio":1.0,
      "short_line_ratio":0.0,
      "char_duplicates_ratio":0.0,
      "list_ratio":0.0,
      "unigram_entropy":4.66911,
      "dup_para_frac":0.0,
      "dup_para_char_frac":0.0,
      "dup_line_frac":0.0,
      "dup_line_char_frac ":0.0,
      "symbol_to_word_ratio_hash":0.0,
      "symbol_to_word_ratio_ellipsis":0.0,
      "non_alpha_words_ratio":0.8685446009389671,
      "token_count":403
   }
}
```

## Document annotation format
The annotation format of the a single document is given below. The `document_id` must correspond to the id of the document in the raw data. Note that `document_id` corresponds to `id` in the raw data. This mismatch should be fixed in the future. As a workaround we can search for the presence of either `document_id` or `id` in the raw data.

The scores are provided by the model that was prompted with the prompt defined by `prompt_name` and `prompt_lang`. Note that the scores are defined as a list, as the model potentially scores the document multiple times to estimate model uncertainty. Further information such as the model generated score explanations and possible errors and the timestamp are added. 

```js
{
   "document_id": "<id>",
   "scores": [1, 2, 2],
   "explanations": ["<explanation 1>", "<explanation 2>", "<explanation 3>"],
   "errors": ["<error 1>", "<error 2>", "<error 3>"],
   "time_stamps": [1, 2, 3],
   "meta_information": {
        "prompt_name": "<prompt_name>",
        "prompt_lang": "<prompt_lang>",
        "model_name": "<model_name>",
    }
}
```

In ml_filter, we implement the annotations by pydantic data classes. The `Annotation` class is defined below and can be found [here](https://github.com/EuroLingua-GPT/ml_filter/blob/annotation_format/src/ml_filter/data_processing/document.py#L47-L66).

```python
class MetaInformation(BaseModel):
    """A class representing the meta information for a given document."""

    prompt_name: str
    prompt_lang: str
    model_name: str

class Annotation(BaseModel):
    """A class representing the output document from the model."""

    document_id: str
    scores: List[float] = []
    explanations: List[str] = []
    errors: List[str] = []
    time_stamps: List[int] = []
    document_processing_status: List[DocumentProcessingStatus] = []
    meta_information: MetaInformation
```

# Data pipelines

There are four main data pipelines in Eurolingua. All of these piplines **never** modify the raw data, as they either generate annotation data or sample from the raw CC corpus. 

1. Prompt-based annotation pipeline: Creates the annotations for the raw data based on a prompt (e.g., fineweb-edu) and a model (e.g. Llama-3.1-70B-Instruct).  
2. Classifier training pipeline: Trains a roberta-like model with either a classification or regression head based on the annotations generated in the prompt-based annotation pipeline.
3. Inference pipeline: Infers the annotations for the raw data using the trained classifier.
4. Sampling pipeline: Creates a subset of the raw data for prompt-based annotations, LLM training and other usecases. 

The following sections describe the data format for each of these pipelines.

## Prompt-based annotation pipeline

The prompt-based pipeline expects the provided raw data to be stored in a folder structure as described above. The output of the pipeline is a set of annotation files, one for each raw JSONL file.

For a given raw JSONL file, the corresponding annotation file must follow the format: `<original_file_name>__annotations_<mdoel_name>_<prompt_name>_<language>.jsonl`,
e.g., `00009__annotations_meta-llama--Llama-3.1-70B-Instruct_fine_web_edu_en.jsonl` for `00009.jsonl`.


A fully annotated CC corpus (i.e., the set of annotation files) must be stored in a folder structure that mirrors the raw data folder structure, such that each annotation file can be mapped back to the raw data file. A simplification of this is to store the data in the same folder structure as the raw data. 
The pipeline creates the mirrored folder structure within the `<output_directory_path>/<experiments_directory>/generated_annotations`. The <output_directory_path> is specified in the pipeline settings, the <experiments_directory> and the `generated_annotations` folder are created automatically be the prompt-based annotation pipeline: e.g.: `data/output/2024-12-06__13-20-48__eabd8b7b/generated_annotations`. 

```
generated_annotations
├── 2014-41
│   ├── bg
│   │   └── 00055__annotations_meta-llama--Llama-3.1-70B-Instruct_fine_web_edu_en.jsonl
│   ├── ca
│   │   ├── 00000__annotations_meta-llama--Llama-3.1-70B-Instruct_fine_web_edu_en.jsonl
│   │   ├── 00035__annotations_meta-llama--Llama-3.1-70B-Instruct_fine_web_edu_en.jsonl
│   │   └── 00040__annotations_meta-llama--Llama-3.1-70B-Instruct_fine_web_edu_en.jsonl
│   └── cs
│       ├── 00001__annotations_meta-llama--Llama-3.1-70B-Instruct_fine_web_edu_en.jsonl
│       └── 00009__annotations_meta-llama--Llama-3.1-70B-Instruct_fine_web_edu_en.jsonl
├── 2015-02

[...]
```

## Classifier training pipeline
The classifier training pipeline requires the root folders of the raw data and the annotations, following the format for raw data and annotations described above. The pipeline itself does not create any new data, but it trains a classifier model that can be used in the inference pipeline.

## Inferencing data

Before running the inference pipeline, the raw data must be tokenized. To achieve this each file is tokenized individually using modalities's tokenization functionality. Tokenization is a two step process involving [indexation](https://github.com/Modalities/modalities?tab=readme-ov-file#raw-training-dataset-indexation) and [tokenization](https://github.com/Modalities/modalities?tab=readme-ov-file#raw-training-dataset-tokenization).  

The result is a folder structure that mirrors the raw data folder structure, with the tokenized data stored in an indexation-optimized format (pbin) that is custom to modalities.

```
├── 2014-41
│   ├── bg
│   │   └── 00055.pbin
│   ├── ca
│   │   ├── 00000.pbin
│   │   ├── 00035.pbin
│   │   └── 00040.pbin
│   └── cs
│       ├── 00001.pbin
│       └── 00009.pbin
├── 2015-02

[...]
```

The output of this pipeline is equivalent to the prompt-based pipeline.

## Sampling raw data
Sampling of raw data (the creation of a CC subset) is required for the prompt-based pipeline and for LLM training. The prompt-based pipeline samples from the CC corpus in a language-stratified fashion, leading to a language-balanced dataset. For other usecases we also support random sampling, where the probability of a document being selected is 1/(total number of documents).

To make the sampling process more efficient, we create a sampling index for each raw JSONL file, as described in the [Modalities documentation](https://github.com/Modalities/modalities?tab=readme-ov-file#raw-training-dataset-indexation). The resulting folder structure is shown below.

```
├── 2014-41
│   ├── bg
│   │   └── 00055.idx
│   ├── ca
│   │   ├── 00000.idx
│   │   ├── 00035.idx
│   │   └── 00040.idx
│   └── cs
│       ├── 00001.idx
│       └── 00009.idx
├── 2015-02

[...]
```
After indexation, we leverage the index files to sample the raw data. The output of the sampling pipeline is a folder structure that mirrors the raw data folder structure, with the sampled JSON documents stored in the JSONL files. Note that due to the nature of sampling, it can happen that some files are not sampled. 
