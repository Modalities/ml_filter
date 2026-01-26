"""Utilities for the data splitter."""

from ml_filter.data_pipelines.ablations.utils import parse_score


def normalize_buckets(buckets):
    """Validate and format bucket ranges."""
    if not buckets:
        raise ValueError("buckets must be a non-empty list of [lower, upper] pairs.")
    specs = []
    for bucket in buckets:
        if not isinstance(bucket, (list, tuple)) or len(bucket) != 2:
            raise ValueError("Each bucket must be a [lower, upper] pair.")
        lower, upper = float(bucket[0]), float(bucket[1])
        if lower >= upper:
            raise ValueError(f"Bucket lower bound must be < upper bound: {lower} >= {upper}")
        specs.append((lower, upper, format(lower, ".15g"), format(upper, ".15g")))
    return specs


def bucket_index(score, bucket_specs):
    """Return the bucket index for a score using [lower, upper) except last bucket inclusive."""
    last_index = len(bucket_specs) - 1
    for index, (lower, upper, _lower_label, _upper_label) in enumerate(bucket_specs):
        if score < lower:
            continue
        if index == last_index:
            if score <= upper:
                return index
            return None
        if score < upper:
            return index
    return None


def average_score(row, score_fields, filepath):
    """Compute the average of the requested score fields."""
    missing_fields = [field for field in score_fields if field not in row]
    if missing_fields:
        raise ValueError(f"Missing score fields {missing_fields} in {filepath}")
    try:
        scores = [parse_score(row.get(field)) for field in score_fields]
    except ValueError as exc:
        raise ValueError(f"Invalid score value in {filepath} for fields {score_fields}") from exc
    return sum(scores) / len(scores)


def validate_bucket_template(template: str) -> None:
    """Ensure the bucket filename template supports required fields."""
    try:
        template.format(lower="0", upper="1", language="lang")
    except KeyError as exc:
        raise ValueError("bucket_filename_template must include {lower}, {upper}, and {language}.") from exc
