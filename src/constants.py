from transformers import (
    BertForSequenceClassification,
    PreTrainedModel,
    XLMRobertaForSequenceClassification,
    XLMRobertaXLForSequenceClassification,
)

# Check
EUROPEAN_LANGUAGES = {
    "sq": "Albanian",
    "hy": "Armenian",
    "eu": "Basque",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "is": "Icelandic",
    "ga": "Irish",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "mt": "Maltese",
    "nb": "Norwegian Bokmål",
    "nn": "Norwegian Nynorsk",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sh": "Serbo-Croation",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "tr": "Turkish",
    "uk": "Ukrainian",
}

DEEPL = "deepl"
OPENAI = "openai"
TARGET_LANGAUGE_PLACEHOLDER = "{##TARGET_LANGUAGE##}"
MODEL_CLASS_MAP: dict[str, type[PreTrainedModel]] = {
    "facebook/xlm-roberta-xl": XLMRobertaXLForSequenceClassification,
    "facebookai/xlm-roberta-base": XLMRobertaForSequenceClassification,
    "facebookai/xlm-roberta-large": XLMRobertaForSequenceClassification,
    "jinaai/jina-embeddings-v3": XLMRobertaForSequenceClassification,
    "snowflake/snowflake-arctic-embed-m": BertForSequenceClassification,
    "snowflake/snowflake-arctic-embed-l": BertForSequenceClassification,
    "snowflake/snowflake-arctic-embed-m-v2.0": BertForSequenceClassification,
    "snowflake/snowflake-arctic-embed-l-v2.0": BertForSequenceClassification,
}
