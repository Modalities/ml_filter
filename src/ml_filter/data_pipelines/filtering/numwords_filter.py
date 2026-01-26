from __future__ import annotations

from datatrove.pipeline.base import PipelineStep


class NumWordsFilter(PipelineStep):
	"""Filter Documents by a minimum word-count on a specific field.

	This is a transforming pipeline step (not a reader): it consumes Documents from
	the previous pipeline step and yields only those where the word count computed
	from `column` is >= `min_num_words`.

	Word counting uses Python's `str.split()`; the column value is coerced to string.
	"""

	name = "Filter_By_NumWords"

	def __init__(self, min_num_words: int, column: str = "text"):
		if min_num_words < 0:
			raise ValueError(f"min_num_words must be >= 0 (got {min_num_words})")
		self._min_num_words = int(min_num_words)
		self._column = column

	def _keep(self, doc) -> bool:
		"""Return True if doc passes the word-count filter."""
		value = None
		if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
			value = doc.metadata.get(self._column)
		if value is None and self._column == "text":
			value = getattr(doc, "text", None)
		if value is None:
			return False
		return len(str(value).split()) >= self._min_num_words

	def __call__(self, data, rank: int = 0, world_size: int = 1):
		"""Datatrove pipeline step.

		Datatrove calls PipelineStep instances as: step(data, rank, world_size).
		`data` is an iterable/generator of Documents.
		"""
		for doc in data:
			if self._keep(doc):
				yield doc

	def run(self, data, rank: int = 0, world_size: int = 1):
		# Some Datatrove versions route through `run`.
		yield from self(data, rank, world_size)

