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
from .generate import generate_full_aar_streaming
from .transcript import load as load_transcript

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = REPO_ROOT / "ui"
DEMO_DIR = REPO_ROOT / "demo_inputs"
CACHE_DIR = REPO_ROOT / "examples" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
            out.append({
                "id": p.stem,
                "title": meta.get("title", p.stem),
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


def _main():
    import uvicorn
    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    _main()
