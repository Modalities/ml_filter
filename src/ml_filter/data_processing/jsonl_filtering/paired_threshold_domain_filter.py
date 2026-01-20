import json
from pathlib import Path
from typing import Literal

from datatrove.io import DataFolder, DataFolderLike
from datatrove.pipeline.base import PipelineStep


class PairedThresholdAndDomainFilter(PipelineStep):
    """Filter Documents by domains using a paired domains JSONL.

    This step expects Documents that include:
      - `metadata` with the text id key (e.g. "document_id" or "id")
      - `metadata.file_relpath` so we can locate the paired domains file

    It loads the paired domains JSONL for each file once and caches an id->domain
    map. Only documents whose domain is in `accepted_domains` are yielded.
    """

    name = "Filter_By_Domain"

    def __init__(
        self,
        domains_data_folder: DataFolderLike,
        accepted_domains: list[str],
        text_jsonl_id_key: str = "document_id",
        domain_jsonl_id_key: str = "document_id",
        domain_jsonl_domain_key: str = "domain",
        compression: Literal["infer", "gzip", "zstd"] | None = None,
    ):
        if not accepted_domains:
            raise ValueError("accepted_domains must contain at least one value")
        self._domains_data_folder = DataFolder(domains_data_folder)
        self._accepted_domains = {str(d) for d in accepted_domains if d is not None}
        self._text_jsonl_id_key = text_jsonl_id_key
        self._domain_jsonl_id_key = domain_jsonl_id_key
        self._domain_jsonl_domain_key = domain_jsonl_domain_key
        self._compression = compression
        self._domain_cache: dict[str, dict[str, str]] = {}

    def _load_domains_for_file(self, relpath: str) -> dict[str, str]:
        if relpath in self._domain_cache:
            return self._domain_cache[relpath]

        domains_abs = Path(str(self._domains_data_folder.path)) / relpath
        if not domains_abs.exists():
            raise FileNotFoundError(
                "Paired domains JSONL file not found for text JSONL. "
                f"relative_path={relpath!r} expected_domains_path={str(domains_abs)!r}"
            )

        domain_map: dict[str, str] = {}
        with self._domains_data_folder.open(relpath, "r", compression=self._compression) as f:
            for li, line in enumerate(f):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"JSON decode error in domains file at {relpath} line={li}: {e}"
                    )
                doc_id = row.get(self._domain_jsonl_id_key)
                if doc_id is None:
                    raise ValueError(
                        f"Missing id key {self._domain_jsonl_id_key!r} in domains file {relpath} line={li}"
                    )
                domain_value = row.get(self._domain_jsonl_domain_key)
                if domain_value is None:
                    raise ValueError(
                        f"Missing domain key {self._domain_jsonl_domain_key!r} in domains file {relpath} line={li}"
                    )
                domain_map[str(doc_id)] = str(domain_value)

        self._domain_cache[relpath] = domain_map
        return domain_map

    def _keep(self, doc) -> bool:
        if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
            return False
        relpath = doc.metadata.get("file_relpath")
        if not relpath:
            raise ValueError("Document metadata missing file_relpath; cannot resolve domains file.")
        doc_id = doc.metadata.get(self._text_jsonl_id_key)
        if doc_id is None:
            raise ValueError(
                f"Document metadata missing id key {self._text_jsonl_id_key!r}; cannot apply domain filter."
            )
        domain_map = self._load_domains_for_file(str(relpath))
        domain_value = domain_map.get(str(doc_id))
        if domain_value is None:
            raise ValueError(
                f"Document id {doc_id!r} not found in domains file {relpath}."
            )
        return domain_value in self._accepted_domains

    def __call__(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            if self._keep(doc):
                yield doc

    def run(self, data, rank: int = 0, world_size: int = 1):
        yield from self(data, rank, world_size)
