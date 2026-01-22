from __future__ import annotations

from datatrove.data import Document

from ml_filter.data_processing.jsonl_filtering.numwords_filter import NumWordsFilter


def test_numwords_filter_uses_metadata_column():
    f = NumWordsFilter(min_num_words=3, column="foo")

    docs = [
        Document(text="ignored", id="a", metadata={"foo": "one two three"}),
        Document(text="ignored", id="b", metadata={"foo": "one"}),
    ]

    kept = list(f(docs, rank=0, world_size=1))
    assert [d.id for d in kept] == ["a"]


def test_numwords_filter_falls_back_to_doc_text_when_column_is_text():
    f = NumWordsFilter(min_num_words=2, column="text")

    docs = [
        Document(text="one two", id="a", metadata={}),
        Document(text="one", id="b", metadata={}),
    ]

    kept = list(f(docs, rank=0, world_size=1))
    assert [d.id for d in kept] == ["a"]


def test_numwords_filter_drops_docs_when_column_missing_and_not_text():
    f = NumWordsFilter(min_num_words=1, column="missing")

    docs = [Document(text="hello", id="a", metadata={})]

    kept = list(f(docs, rank=0, world_size=1))
    assert kept == []
