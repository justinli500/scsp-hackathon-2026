"""Semantic search over the doctrine index. Returns chunks with the metadata
needed to render an inline citation and verify it (chunk_id round-trip)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REPO_ROOT / "corpus" / "index"
COLLECTION = "doctrine"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Hit:
    chunk_id: str
    source_title: str
    source_stem: str
    doc_type: str
    page_start: int
    page_end: int
    text: str
    distance: float

    @property
    def citation(self) -> str:
        if self.page_start == self.page_end:
            return f"[{self.source_title}, p. {self.page_start}]"
        return f"[{self.source_title}, pp. {self.page_start}–{self.page_end}]"


_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        _collection = client.get_collection(name=COLLECTION, embedding_function=embed_fn)
    return _collection


def search(query: str, k: int = 5, doc_types: Iterable[str] | None = None) -> list[Hit]:
    coll = _get_collection()
    where = {"doc_type": {"$in": list(doc_types)}} if doc_types else None
    res = coll.query(query_texts=[query], n_results=k, where=where)
    hits: list[Hit] = []
    if not res["ids"] or not res["ids"][0]:
        return hits
    for i, cid in enumerate(res["ids"][0]):
        m = res["metadatas"][0][i]
        hits.append(Hit(
            chunk_id=cid,
            source_title=m["source_title"],
            source_stem=m["source_stem"],
            doc_type=m["doc_type"],
            page_start=int(m["page_start"]),
            page_end=int(m["page_end"]),
            text=res["documents"][0][i],
            distance=float(res["distances"][0][i]) if res.get("distances") else 0.0,
        ))
    return hits


def get_by_id(chunk_id: str) -> Hit | None:
    """Used by the citation-verification gate. Returns None if the chunk doesn't exist."""
    coll = _get_collection()
    res = coll.get(ids=[chunk_id])
    if not res["ids"]:
        return None
    m = res["metadatas"][0]
    return Hit(
        chunk_id=res["ids"][0],
        source_title=m["source_title"],
        source_stem=m["source_stem"],
        doc_type=m["doc_type"],
        page_start=int(m["page_start"]),
        page_end=int(m["page_end"]),
        text=res["documents"][0],
        distance=0.0,
    )


def _cli():
    import sys
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = [
            "what does doctrine say about branches and sequels",
            "after action review sustain and improve format",
            "adversary anti-access area-denial Chinese tactics",
        ]
    for q in queries:
        print(f"\n=== query: {q!r} ===")
        for h in search(q, k=5):
            snippet = h.text[:200].replace("\n", " ")
            print(f"  {h.citation}  dist={h.distance:.3f}  id={h.chunk_id}")
            print(f"    {snippet}…")


if __name__ == "__main__":
    _cli()
