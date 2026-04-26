"""Ingest doctrine PDFs in corpus/raw/ -> per-page text -> token-bounded chunks
-> embeddings -> Chroma index at corpus/index/. Also dumps per-doc chunks as
JSON to corpus/chunks/ for inspection — citation traceability starts here.

Run:  python -m src.ingest
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pdfplumber
import tiktoken
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "corpus" / "raw"
CHUNKS_DIR = REPO_ROOT / "corpus" / "chunks"
INDEX_DIR = REPO_ROOT / "corpus" / "index"

CHUNK_TOKENS = 500
CHUNK_OVERLAP = 75
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "doctrine"

# Maps stem -> (display title, doc_type). Display title is what shows in citations.
DOC_REGISTRY: dict[str, tuple[str, str]] = {
    "jp5_0": ("JP 5-0", "JP"),
    "jp3_0": ("JP 3-0", "JP"),
    "fm3_0": ("FM 3-0", "FM"),
    "fm5_0": ("FM 5-0", "FM"),
    "atp7_100_3": ("ATP 7-100.3", "ATP"),
    "aar_leaders_guide_2013": ("AAR Leader's Guide (2013)", "Guide"),
}


@dataclass
class Chunk:
    chunk_id: str           # stable id used as citation key, e.g. "jp5_0:p42:c0"
    source_stem: str        # filename stem
    source_title: str       # display title for citations
    doc_type: str           # JP / FM / ATP / Guide
    page_start: int         # 1-indexed
    page_end: int
    text: str
    token_count: int


def _tokenizer():
    return tiktoken.get_encoding("cl100k_base")


def _doc_meta(stem: str) -> tuple[str, str]:
    if stem in DOC_REGISTRY:
        return DOC_REGISTRY[stem]
    # Unknown PDF — synthesize a title from the stem so it still flows.
    title = stem.replace("_", " ").upper()
    return title, "Other"


def _normalize(text: str) -> str:
    # PDFs from Government Printing Office often have running headers/footers like
    # "JP 5-0" on every page — those add noise to retrieval. Strip lines that look
    # like page-marker artifacts and collapse whitespace.
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append("")
            continue
        # Drop common page-marker patterns: "II-12", "Chapter II", bare page numbers
        if re.fullmatch(r"[IVXLCDM]+-\d+", s):
            continue
        if re.fullmatch(r"\d{1,4}", s):
            continue
        out.append(ln)
    text = "\n".join(out)
    # Collapse runs of blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Returns [(page_number_1indexed, text), ...]. Skips pages with no text."""
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            txt = _normalize(txt)
            if txt:
                pages.append((i, txt))
    return pages


def _chunk_pages(
    pages: list[tuple[int, str]],
    stem: str,
    title: str,
    doc_type: str,
) -> list[Chunk]:
    """Chunk by token count with overlap. Each chunk records page_start/page_end
    so we can render `[Title, p. 42–44]` citations honestly."""
    enc = _tokenizer()
    # Token-tag every page so we can find which page boundary each chunk straddles.
    tagged: list[tuple[int, list[int]]] = [(p, enc.encode(t)) for p, t in pages]

    # Flatten with page bookkeeping.
    flat_tokens: list[int] = []
    page_at: list[int] = []  # parallel array: which page each token came from
    for page_no, toks in tagged:
        flat_tokens.extend(toks)
        page_at.extend([page_no] * len(toks))

    chunks: list[Chunk] = []
    if not flat_tokens:
        return chunks

    step = CHUNK_TOKENS - CHUNK_OVERLAP
    n = len(flat_tokens)
    cid = 0
    start = 0
    while start < n:
        end = min(start + CHUNK_TOKENS, n)
        slice_tokens = flat_tokens[start:end]
        text = enc.decode(slice_tokens).strip()
        if text:
            page_start = page_at[start]
            page_end = page_at[end - 1]
            chunks.append(Chunk(
                chunk_id=f"{stem}:p{page_start}-{page_end}:c{cid}",
                source_stem=stem,
                source_title=title,
                doc_type=doc_type,
                page_start=page_start,
                page_end=page_end,
                text=text,
                token_count=len(slice_tokens),
            ))
            cid += 1
        if end == n:
            break
        start += step
    return chunks


def _save_chunks_json(stem: str, chunks: list[Chunk]) -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out = CHUNKS_DIR / f"{stem}.json"
    out.write_text(json.dumps([asdict(c) for c in chunks], indent=2))


def ingest_all(rebuild: bool = True) -> None:
    pdfs = sorted(p for p in RAW_DIR.glob("*.pdf") if p.stat().st_size > 0)
    if not pdfs:
        raise SystemExit(f"No non-empty PDFs in {RAW_DIR}. Run scripts/download_artifacts.sh first.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=COLLECTION, embedding_function=embed_fn)

    total_chunks = 0
    for pdf in pdfs:
        stem = pdf.stem
        title, doc_type = _doc_meta(stem)
        print(f"\n[ingest] {pdf.name}  ({title}, {doc_type})")
        pages = _extract_pages(pdf)
        print(f"  pages with text: {len(pages)}")
        if not pages:
            print(f"  ! no extractable text — skipping")
            continue
        chunks = _chunk_pages(pages, stem, title, doc_type)
        print(f"  chunks: {len(chunks)}")
        _save_chunks_json(stem, chunks)

        # Add to chroma in batches (chroma has a per-call limit).
        BATCH = 256
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i:i + BATCH]
            collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[{
                    "source_stem": c.source_stem,
                    "source_title": c.source_title,
                    "doc_type": c.doc_type,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "token_count": c.token_count,
                } for c in batch],
            )
        total_chunks += len(chunks)

    print(f"\n[ingest] done. {total_chunks} chunks indexed in {INDEX_DIR}")


if __name__ == "__main__":
    ingest_all()
