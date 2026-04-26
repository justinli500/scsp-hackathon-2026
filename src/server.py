"""FastAPI app exposing the AAR generator over the web.

Endpoints:
  GET  /                       — single-page UI (ui/index.html)
  GET  /api/demos              — list available demo transcripts
  POST /api/generate           — SSE stream of AAR generation events
  GET  /api/chunk/{chunk_id}   — returns a doctrine chunk for the citation modal
  GET  /api/health             — sanity check (corpus loaded, key present)

Run:  python -m src.server     (or: uvicorn src.server:app --reload)
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import retrieval
from .analyze import analyze, section_markdowns_from_events
from .generate import generate_full_aar_streaming
from .ingest import DOC_REGISTRY
from .transcript import load as load_transcript

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = REPO_ROOT / "ui"
DEMO_DIR = REPO_ROOT / "demo_inputs"
CACHE_DIR = REPO_ROOT / "examples" / "cache"
EXAMPLES_DIR = REPO_ROOT / "examples"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Section-header text in saved markdown AARs maps back to internal section_ids.
_MD_HEADER_TO_SECTION = {
    "1. Introduction and Rules": "intro_rules",
    "1. Introduction & Rules": "intro_rules",
    "2. Review of Objectives": "objectives_review",
    "3. Orientation": "orientation",
    "4. Summary of Recent Events": "what_happened",
    "5. Discussion of Key Issues": "key_issues",
    "6. Force Protection and Optional Issues": "force_protection",
    "6. Force Protection / Optional Issues": "force_protection",
    "7. Closing Comments": "closing",
    "Lessons Identified": "lessons_identified",
}


_SECTION_ORDER_FOR_BOOTSTRAP = [
    "intro_rules", "objectives_review", "orientation", "what_happened",
    "key_issues", "force_protection", "closing", "lessons_identified",
]


def _bootstrap_cache_from_saved_markdown(demo_id: str, transcript_title: str) -> None:
    """If a saved AAR markdown exists for this demo but no event-stream cache,
    synthesize a cache file so Replay and analysis both work without re-running
    generation. Tokens are split coarsely (one event per line) for replay
    smoothness; full markdown is preserved on each section_end."""
    cache_p = _cache_path(demo_id)
    if cache_p.exists():
        return
    section_md = _section_md_from_saved_markdown(demo_id)
    if not section_md:
        return

    events: list[dict] = [{"type": "aar_start", "title": transcript_title}]
    # Use the canonical section order; only emit sections we actually parsed.
    for sid in _SECTION_ORDER_FOR_BOOTSTRAP:
        md = section_md.get(sid)
        if not md:
            continue
        # The downstream UI renderer skips the first H2 (assuming the model
        # leads with one). Saved markdown has no H2 within the body, so prepend
        # a placeholder to keep the renderer happy.
        body = f"## {sid}\n\n{md.rstrip()}"
        events.append({"type": "section_start", "section_id": sid})
        # Emit roughly per-line token deltas so replay animates rather than dumps.
        for line in body.split("\n"):
            events.append({"type": "token", "section_id": sid, "text": line + "\n"})
        # Re-extract chunk_ids for the cited list.
        from .analyze import _extract_chunk_ids
        cids = _extract_chunk_ids(body)
        events.append({
            "type": "section_end",
            "section_id": sid,
            "cited_chunk_ids": cids,
            "markdown": body,
        })
    events.append({"type": "aar_end"})

    payload = {
        "demo_id": demo_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "had_section_error": False,
        "events": events,
        "_source": "bootstrapped_from_saved_markdown",
    }
    try:
        cache_p.write_text(json.dumps(payload))
    except Exception:
        pass


def _section_md_from_saved_markdown(demo_id: str) -> dict[str, str] | None:
    """Fallback: a previously saved AAR may exist as a flat markdown file in
    examples/. Parse it back into a section_id -> markdown mapping so the
    analysis pipeline can run without an SSE event cache."""
    candidates = [
        EXAMPLES_DIR / f"{demo_id}_aar.md",
        EXAMPLES_DIR / f"{demo_id}.md",
    ]
    md_path = next((p for p in candidates if p.exists()), None)
    if md_path is None:
        return None
    raw = md_path.read_text()
    # Split on H2 headers; keep heading text with body.
    chunks = raw.split("\n## ")
    out: dict[str, str] = {}
    for chunk in chunks[1:]:  # first chunk is the H1 preamble
        if "\n" not in chunk:
            continue
        header, body = chunk.split("\n", 1)
        sid = _MD_HEADER_TO_SECTION.get(header.strip())
        if sid:
            out[sid] = body
    return out or None

# Replay token delay. Real generation streams at variable pace; replay uses a
# small constant delay so the UI animation still reads as "watching it think."
REPLAY_TOKEN_DELAY_S = 0.005

app = FastAPI(title="Doctrinal AAR Generator")


@app.get("/api/health")
def health():
    raw_pdfs = sorted(p.name for p in (REPO_ROOT / "corpus" / "raw").glob("*.pdf") if p.stat().st_size > 0)
    chunks = sorted(p.name for p in (REPO_ROOT / "corpus" / "chunks").glob("*.json"))
    return {
        "ok": True,
        "anthropic_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
        "doctrine_pdfs": raw_pdfs,
        "doctrine_chunk_files": chunks,
        "demos": _list_demos(),
    }


def _cache_path(demo_id: str) -> Path:
    return CACHE_DIR / f"{demo_id}.json"


def _cache_meta(demo_id: str) -> dict | None:
    p = _cache_path(demo_id)
    if not p.exists():
        return None
    try:
        st = p.stat()
        with p.open() as f:
            head = json.load(f)
        return {
            "generated_at": head.get("generated_at"),
            "n_events": len(head.get("events", [])),
            "size_bytes": st.st_size,
        }
    except Exception:
        return None


def _list_demos() -> list[dict]:
    out: list[dict] = []
    for p in sorted(DEMO_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            meta = data.get("meta", {})
            title = meta.get("title", p.stem)
            # Seed an event-stream cache from any saved markdown so the UI's
            # Replay + Analyze work on previously-saved AARs.
            _bootstrap_cache_from_saved_markdown(p.stem, title)
            out.append({
                "id": p.stem,
                "title": title,
                "scenario": meta.get("scenario", ""),
                "source": meta.get("source", ""),
                "n_entries": len(data.get("entries", [])),
                "cache": _cache_meta(p.stem),
            })
        except Exception:
            continue
    return out


@app.get("/api/demos")
def demos():
    return _list_demos()


@app.get("/api/doctrine")
def doctrine_library():
    """List the doctrine PDFs available for direct viewing. Each entry carries
    the canonical citation title plus a URL the UI can link to."""
    raw_dir = REPO_ROOT / "corpus" / "raw"
    out: list[dict] = []
    for pdf in sorted(raw_dir.glob("*.pdf")):
        stem = pdf.stem
        title, doc_type = DOC_REGISTRY.get(stem, (stem, "Doc"))
        out.append({
            "stem": stem,
            "title": title,
            "doc_type": doc_type,
            "filename": pdf.name,
            "url": f"/doctrine/{pdf.name}",
            "size_bytes": pdf.stat().st_size,
        })
    return out


@app.get("/api/chunk/{chunk_id:path}")
def chunk(chunk_id: str):
    h = retrieval.get_by_id(chunk_id)
    if h is None:
        raise HTTPException(status_code=404, detail=f"chunk_id not found: {chunk_id}")
    return {
        "chunk_id": h.chunk_id,
        "source_title": h.source_title,
        "source_stem": h.source_stem,
        "doc_type": h.doc_type,
        "page_start": h.page_start,
        "page_end": h.page_end,
        "text": h.text,
        "citation": h.citation,
    }


@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    demo_id = body.get("demo_id")
    if not demo_id:
        raise HTTPException(status_code=400, detail="demo_id required")
    transcript_path = DEMO_DIR / f"{demo_id}.json"
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail=f"demo not found: {demo_id}")
    transcript = load_transcript(transcript_path)

    def event_stream():
        captured: list[dict] = []
        had_section_error = False
        try:
            for ev in generate_full_aar_streaming(transcript):
                captured.append(ev)
                if ev.get("type") == "section_error":
                    had_section_error = True
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            err = {"type": "fatal", "error": str(e), "error_class": e.__class__.__name__}
            captured.append(err)
            yield f"data: {json.dumps(err)}\n\n"
            return  # don't write a fatal-only stream to cache
        # Persist the event stream so the UI can replay this run later. We
        # save even when section_error fired — the failed-section is part of
        # the demo's "the gate caught it" story, not an error to hide.
        try:
            payload = {
                "demo_id": demo_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "had_section_error": had_section_error,
                "events": captured,
            }
            _cache_path(demo_id).write_text(json.dumps(payload))
        except Exception:
            # Caching is best-effort; never let a write failure disrupt the user's stream.
            pass

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.post("/api/replay")
async def replay(request: Request):
    """Stream a previously-cached AAR back as SSE so the UI animation still
    reads as live generation. Token deltas are replayed with a small constant
    delay; structural events (section_start/end, errors) are emitted instantly."""
    body = await request.json()
    demo_id = body.get("demo_id")
    if not demo_id:
        raise HTTPException(status_code=400, detail="demo_id required")
    p = _cache_path(demo_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"no cached AAR for {demo_id}")
    payload = json.loads(p.read_text())
    events = payload.get("events", [])

    def event_stream():
        for ev in events:
            yield f"data: {json.dumps(ev)}\n\n"
            if ev.get("type") == "token":
                time.sleep(REPLAY_TOKEN_DELAY_S)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.delete("/api/cache/{demo_id}")
def cache_delete(demo_id: str):
    p = _cache_path(demo_id)
    if p.exists():
        p.unlink()
    a = _analysis_path(demo_id)
    if a.exists():
        a.unlink()
    return {"ok": True, "demo_id": demo_id}


def _analysis_path(demo_id: str) -> Path:
    return CACHE_DIR / f"{demo_id}.analysis.json"


@app.post("/api/analyze")
async def analyze_endpoint(request: Request):
    """Run (or return cached) post-AAR analysis: timeline, drill traces, eval,
    gaps. Requires a cached AAR for the demo (run /api/generate first, or have
    a previously saved run). Pass {"force": true} to bypass the cache."""
    body = await request.json()
    demo_id = body.get("demo_id")
    force = bool(body.get("force"))
    if not demo_id:
        raise HTTPException(status_code=400, detail="demo_id required")

    transcript_path = DEMO_DIR / f"{demo_id}.json"
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail=f"demo not found: {demo_id}")
    analysis_p = _analysis_path(demo_id)
    if analysis_p.exists() and not force:
        try:
            return JSONResponse(json.loads(analysis_p.read_text()))
        except Exception:
            pass  # fall through and re-run

    transcript = load_transcript(transcript_path)

    # Prefer a fresh event-stream cache, but fall back to a saved markdown AAR
    # in examples/ so analysis works on previously-saved runs.
    cache_p = _cache_path(demo_id)
    section_md: dict[str, str] = {}
    if cache_p.exists():
        cached = json.loads(cache_p.read_text())
        section_md = section_markdowns_from_events(cached.get("events", []))
    if not section_md:
        fallback = _section_md_from_saved_markdown(demo_id)
        if fallback:
            section_md = fallback
    if not section_md:
        raise HTTPException(
            status_code=409,
            detail=(
                f"no cached AAR for {demo_id}; generate one or place a saved "
                f"markdown at examples/{demo_id}_aar.md."
            ),
        )

    result = analyze(demo_id, transcript, section_md)
    payload = result.to_json()
    try:
        analysis_p.write_text(json.dumps(payload))
    except Exception:
        pass
    return JSONResponse(payload)


@app.delete("/api/analyze/{demo_id}")
def analyze_delete(demo_id: str):
    p = _analysis_path(demo_id)
    if p.exists():
        p.unlink()
    return {"ok": True, "demo_id": demo_id}


@app.get("/")
def index():
    idx = UI_DIR / "index.html"
    if not idx.exists():
        return JSONResponse({"error": "ui/index.html not found"}, status_code=500)
    return FileResponse(idx)


# Serve any additional UI assets if they appear later.
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

# Serve the source doctrine PDFs so the reference-library modal can link
# directly to them. Read-only; no upload path.
DOCTRINE_RAW_DIR = REPO_ROOT / "corpus" / "raw"
if DOCTRINE_RAW_DIR.exists():
    app.mount("/doctrine", StaticFiles(directory=str(DOCTRINE_RAW_DIR)), name="doctrine")


def _main():
    import uvicorn
    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    _main()
