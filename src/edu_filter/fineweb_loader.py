from datasets import load_dataset


if __name__=="__main__":

    fineweb_data = load_dataset("HuggingFaceFW/fineweb-edu-llama3-annotations", split="train", cache_dir="../../../fineweb_data")
    
    five_dataset_ = fineweb_data.filter(lambda example: example['score'] == 5)
    three_dataset_ = fineweb_data.filter(lambda example: example['score'] == 3)
    two_dataset_ = fineweb_data.filter(lambda example: example['score'] == 2)

     # Save each filtered dataset into a separate jsonl file
    five_dataset_.to_json("five_dataset.jsonl")
    three_dataset_.to_json("three_dataset.jsonl")
    two_dataset_.to_json("two_dataset.jsonl")
