import argparse
import json
from pathlib import Path

from tabulate import tabulate

# Pricing data (prices per 1M tokens)
PRICES = {
    "batched": {
        "gpt-5": {"input": 0.625, "cached_input": 0.0625, "output": 5.00},
        "gpt-5-mini": {"input": 0.125, "cached_input": 0.0125, "output": 1.00},
        "gpt-5-nano": {"input": 0.025, "cached_input": 0.0025, "output": 0.20},
    },
    "not batched": {
        "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
        "gpt-5-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        "gpt-5-codex": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
        "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-2024-05-13": {"input": 5.00, "cached_input": None, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
        "gpt-realtime": {"input": 4.00, "cached_input": 0.40, "output": 16.00},
        "gpt-4o-realtime-preview": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
        "gpt-4o-mini-realtime-preview": {"input": 0.60, "cached_input": 0.30, "output": 2.40},
        "gpt-audio": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-4o-audio-preview": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-4o-mini-audio-preview": {"input": 0.15, "cached_input": None, "output": 0.60},
        "o1": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
        "o1-pro": {"input": 150.00, "cached_input": None, "output": 600.00},
        "o3-pro": {"input": 20.00, "cached_input": None, "output": 80.00},
        "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "o3-deep-research": {"input": 10.00, "cached_input": 2.50, "output": 40.00},
        "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
        "o4-mini-deep-research": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
        "o1-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
    },
}


def get_model_price_key(model_name, price_keys):
    """
    Finds the best matching price key for a given model name by finding the
    longest prefix match. This correctly distinguishes 'gpt-5' from 'gpt-5-mini'.
    """
    matching_keys = [key for key in price_keys if model_name.startswith(key)]
    if not matching_keys:
        return None
    return max(matching_keys, key=len)


def calculate_cost(model, usage, cost_plan):
    """Calculates the cost for a single API call based on token usage."""
    plan_prices = PRICES.get(cost_plan, {})

    # Use the robust matching function to find the correct price key
    model_key = get_model_price_key(model, plan_prices.keys())

    if not model_key:
        return 0.0

    model_prices = plan_prices.get(model_key)
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    input_tokens = prompt_tokens - cached_tokens

    cost = input_tokens * model_prices["input"]
    if model_prices.get("cached_input") is not None:
        cost += cached_tokens * model_prices["cached_input"]
    cost += completion_tokens * model_prices["output"]

    return cost / 1_000_000


def find_and_process_files(root_dir, output_file):
    """Finds all 'batch_results.jsonl' files and processes their content."""
    all_results = []
    total_cost_batched = 0.0
    total_cost_not_batched = 0.0

    search_path = Path(root_dir)
    if not search_path.is_dir():
        print(f"Error: Directory not found at '{root_dir}'")
        return

    for file_path in search_path.rglob("batch_results.jsonl"):
        parent_folder = file_path.parent.name
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    response_body = data.get("response", {}).get("body", {})
                    if not response_body:
                        continue

                    usage = response_body.get("usage")
                    model = response_body.get("model")

                    if not usage or not model:
                        continue

                    cost_batched = calculate_cost(model, usage, "batched")
                    cost_not_batched = calculate_cost(model, usage, "not batched")

                    total_cost_batched += cost_batched
                    total_cost_not_batched += cost_not_batched

                    all_results.append(
                        [
                            parent_folder,
                            model,
                            usage.get("prompt_tokens", 0),
                            usage.get("completion_tokens", 0),
                            usage.get("total_tokens", 0),
                            f"${cost_batched:.6f}",
                            f"${cost_not_batched:.6f}",
                        ]
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not process line {i} in {file_path}. Error: {e}")

    generate_markdown_report(all_results, total_cost_batched, total_cost_not_batched, output_file)


def generate_markdown_report(results, total_cost_batched, total_cost_not_batched, output_file):
    """Generates and saves the final report as a markdown file."""
    md_content = ["# 📊 Cost Analysis Report: Batched vs. Not Batched Plans", "\n---"]

    for plan_name in ["batched", "not batched"]:
        md_content.append(f"## Price Table: {plan_name.title()} (per 1M tokens)")
        price_headers = ["Model", "Input", "Cached Input", "Output"]
        price_data = [
            [
                model,
                f"${d['input']}",
                f"${d['cached_input']}" if d.get("cached_input") is not None else "N/A",
                f"${d['output']}",
            ]
            for model, d in PRICES[plan_name].items()
        ]
        md_content.append(tabulate(price_data, headers=price_headers, tablefmt="pipe"))
        md_content.append("\n---")

    md_content.append("## Usage and Cost Comparison per Request")
    if not results:
        md_content.append("No 'batch_results.jsonl' files with valid data found.")
    else:
        headers = [
            "Parent Folder",
            "Model",
            "Prompt Tokens",
            "Completion Tokens",
            "Total Tokens",
            "Cost (Batched)",
            "Cost (Not Batched)",
        ]
        md_content.append(tabulate(results, headers=headers, tablefmt="pipe"))

    md_content.append("\n---")
    md_content.append("## 💰 Grand Totals")
    md_content.append(f"- **Batched Plan Total:** `${total_cost_batched:.4f}`")
    md_content.append(f"- **Not Batched Plan Total:** `${total_cost_not_batched:.4f}`")

    try:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))
        print(f"✅ Report successfully saved to '{output_file}'")
    except IOError as e:
        print(f"❌ Error: Could not write report to '{output_file}'. Reason: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process 'batch_results.jsonl' files to generate a markdown cost comparison report."
    )
    parser.add_argument("root_directory", type=str, help="The root directory to search recursively.")
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="cost_comparison_report.md",
        help="Path to save the markdown report (default: cost_comparison_report.md).",
    )

    args = parser.parse_args()

    find_and_process_files(args.root_directory, args.output_file)
