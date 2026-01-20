from __future__ import annotations

from typing import Iterable

from datatrove.pipeline.base import PipelineStep


class DomainFilter(PipelineStep):
    """Filter documents based on a domain value stored in metadata."""

    name = "Filter_By_Domain"

    def __init__(self, accepted_domains: Iterable[str], domain_key: str = "domain"):
        self._accepted_domains = {str(d) for d in accepted_domains if d is not None}
        if not self._accepted_domains:
            raise ValueError("accepted_domains must contain at least one value")
        self._domain_key = domain_key

    def __call__(self, data, rank: int = 0, world_size: int = 1):  # type: ignore[override]
        for doc in data:
            domain_value = doc.metadata.get(self._domain_key) if doc.metadata else None
            if domain_value is None:
                raise ValueError(f"Missing domain key {self._domain_key!r} in document metadata")
            if str(domain_value) in self._accepted_domains:
                yield doc

    def run(self, data, rank: int = 0, world_size: int = 1):
        yield from self(data, rank, world_size)
