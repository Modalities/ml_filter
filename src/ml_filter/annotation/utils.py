# ---------------------------------------------------------------------------
# Precision mappings (duplicated locally to avoid cross-module dependency for
# simple pipeline configuration & validation).
# ---------------------------------------------------------------------------

# Unified model compute dtype mapping (torch only)
from typing import Any
import torch, numpy as np  # local import for dtypes
MODEL_TORCH_DTYPE_MAPPING: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Embedding storage dtype (numpy). Note: bfloat16 stored as float32 due to limited numpy support.
EMBED_NUMPY_DTYPE_MAPPING: dict[str, np.dtype] = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": np.float32,
}

EMBED_TORCH_DTYPE_MAPPING: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Label storage dtype mapping (extend as needed)
LABEL_NUMPY_DTYPE_MAPPING: dict[str, np.dtype] = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": np.float32,  # degrade
}

def resolve_output_dtype(schema: Any, pipeline: str) -> dict[str, Any]:
    """Resolve and validate precision dtypes based on pipeline context.

    Args:
        schema: dict with optional keys: model_dtype, embedding_dtype, label_dtype
        pipeline: one of {"embedding_pipeline", "annotation_pipeline"}

    Behavior:
      - model_dtype always resolved to a torch dtype (MODEL_TORCH_DTYPE_MAPPING)
      - embedding_dtype mapping selection:
            embedding_pipeline   -> EMBED_NUMPY_DTYPE_MAPPING (numpy dtype, bfloat16 degraded)
            annotation_pipeline  -> EMBED_TORCH_DTYPE_MAPPING (torch dtype)
      - label_dtype always resolved via LABEL_NUMPY_DTYPE_MAPPING (storage dtype) if provided

    Defaults when schema is not a dict:
        model = torch.bfloat16, embedding = np.float32 (embedding_pipeline) or torch.bfloat16 (annotation_pipeline), label = np.float16
    """
    if pipeline not in {"embedding_pipeline", "annotation_pipeline"}:
        raise ValueError("pipeline must be 'embedding_pipeline' or 'annotation_pipeline'")

    if not isinstance(schema, dict):
        return {
            'model_dtype': torch.bfloat16,
            'embedding_dtype': np.float32 if pipeline == "embedding_pipeline" else torch.bfloat16,
            'label_dtype': np.float16,
        }

    model_raw = str(schema.get('model_dtype')).lower()
    emb_raw = str(schema.get('embedding_dtype')).lower()
    label_raw = str(schema.get('label_dtype')).lower()

    if model_raw not in MODEL_TORCH_DTYPE_MAPPING:
        raise ValueError(f"Unsupported model dtype '{model_raw}'. Allowed: {', '.join(MODEL_TORCH_DTYPE_MAPPING.keys())}")

    # Choose embedding mapping based on pipeline
    if pipeline == 'embedding_pipeline':
        emb_map = EMBED_NUMPY_DTYPE_MAPPING
    else:  # annotation_pipeline
        emb_map = EMBED_TORCH_DTYPE_MAPPING

    if emb_raw not in emb_map:
        raise ValueError(
            f"Unsupported embedding dtype '{emb_raw}' for {pipeline}. Allowed: {', '.join(emb_map.keys())}"
        )

    # Label dtype optional
    if label_raw not in LABEL_NUMPY_DTYPE_MAPPING:
        raise ValueError(
            f"Unsupported label dtype '{label_raw}'. Allowed: {', '.join(LABEL_NUMPY_DTYPE_MAPPING.keys())}"
        )

    return {
        'model_dtype': MODEL_TORCH_DTYPE_MAPPING[model_raw],
        'embedding_dtype': emb_map[emb_raw],
        'label_dtype': LABEL_NUMPY_DTYPE_MAPPING[label_raw],
    }