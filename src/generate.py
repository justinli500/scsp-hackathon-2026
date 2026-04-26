"""AAR generation with citation verification gate.

Pipeline (per section):
  1. Build a query string from transcript + section purpose.
  2. Retrieve top-k doctrine chunks.
  3. Send transcript + chunks to claude-opus-4-7 with the section prompt.
  4. Parse {{cite:CHUNK_ID}} tags from the LLM output.
  5. HARD GATE: every cited chunk_id must be resolvable via retrieval.get_by_id.
     If any can't be resolved, raise CitationError. No silent fallback.
  6. Return Markdown with citations preserved (chunk_ids retained inline; the
     UI renderer resolves them to clickable links).

Run:  python -m src.generate demo_inputs/csis_first_battle.json
      python -m src.generate demo_inputs/csis_first_battle.json --full
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from anthropic import Anthropic, APIConnectionError, APIStatusError

from . import retrieval
from .transcript import Transcript, load as load_transcript


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "prompts"
SECTION_PROMPTS_DIR = PROMPTS_DIR / "section_prompts"

MODEL = "claude-opus-4-7"
MAX_TOKENS = 4000
RETRIEVAL_K = 8

# Order of sections in the rendered AAR. lessons_identified runs as a final
# pass after the seven core sections.
SECTION_ORDER = [
    "intro_rules",
    "objectives_review",
    "orientation",
    "what_happened",
    "key_issues",
    "force_protection",
    "closing",
    "lessons_identified",
]

CITE_TAG = re.compile(r"\{\{\s*cite:([^}]+?)\s*\}\s*\}")


class CitationError(RuntimeError):
    """Raised when generation produces a citation that does not resolve to a real chunk."""


@dataclass
class GeneratedSection:
    section_id: str
    markdown: str
    cited_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class FullAAR:
    transcript_title: str
    sections: list[GeneratedSection] = field(default_factory=list)

    def render(self) -> str:
        head = f"# After-Action Review — {self.transcript_title}\n\n"
        return head + "\n\n".join(s.markdown.strip() for s in self.sections) + "\n"


def _read(p: Path) -> str:
    return p.read_text()


def _build_query(transcript: Transcript, section_id: str) -> str:
    """Combine the section purpose with scenario-specific terms to bias retrieval."""
    base = {
        "what_happened": "after action review summary of recent events branches sequels main effort",
        "objectives_review": "after action review of objectives mission essential tasks",
        "key_issues": "after action review sustain improve discussion of key issues",
        "lessons_identified": "lessons identified observation pipeline NATO JLLIS",
        "orientation": "AAR orientation site setup observer coach trainer",
        "intro_rules": "AAR introduction ground rules participation discussion not critique",
        "force_protection": "force protection casualty risk operational risk management",
        "closing": "AAR closing comments lessons identified leader of unit training management",
    }
    section_query = base.get(section_id, section_id.replace("_", " "))
    scenario_query = (transcript.scenario or "")[:300]
    return f"{section_query}. {scenario_query}"


def _format_chunks(hits: list[retrieval.Hit]) -> str:
    lines = ["RETRIEVED CHUNKS (cite using these chunk_ids and these only):", ""]
    for h in hits:
        lines.append(f"chunk_id: {h.chunk_id}")
        lines.append(f"citation: {h.citation}")
        lines.append("text:")
        lines.append(h.text.strip())
        lines.append("---")
    return "\n".join(lines)


def _verify_citations(markdown: str) -> list[str]:
    """Parse all {{cite:...}} tags. Verify each resolves. Raise on any miss."""
    cited = CITE_TAG.findall(markdown)
    bad: list[str] = []
    for cid in cited:
        if retrieval.get_by_id(cid) is None:
            bad.append(cid)
    if bad:
        raise CitationError(
            f"Generation produced {len(bad)} unresolvable citation(s): {bad}. "
            f"This is a hard gate — refusing to publish AAR with hallucinated citations."
        )
    return cited


def _section_payload(transcript: Transcript, section_id: str) -> tuple[str, str]:
    """Returns (system, user) message pair for this section."""
    section_prompt_path = SECTION_PROMPTS_DIR / f"{section_id}.txt"
    if not section_prompt_path.exists():
        raise FileNotFoundError(f"No prompt for section {section_id!r} at {section_prompt_path}")

    system = _read(PROMPTS_DIR / "aar_system.txt") + "\n\n" + _read(PROMPTS_DIR / "citation_format.txt")
    section_instructions = _read(section_prompt_path)

    query = _build_query(transcript, section_id)
    hits = retrieval.search(query, k=RETRIEVAL_K)
    if not hits:
        raise RuntimeError("No chunks retrieved — index may be empty or query is off-topic.")

    user = (
        f"{section_instructions}\n\n"
        f"---\n\nTRANSCRIPT:\n\n{transcript.render_for_prompt()}\n\n"
        f"---\n\n{_format_chunks(hits)}"
    )
    return system, user


def _client() -> Anthropic:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Export it or add to a .env loaded before running."
        )
    return Anthropic()


def generate_section(
    transcript: Transcript,
    section_id: str,
    *,
    attempts: int = 3,
    citation_retries: int = 2,
) -> GeneratedSection:
    """Synchronous wrapper that internally uses streaming. Streaming keeps the
    connection healthier for ~30-60s long outputs.

    Two retry budgets:
      - `attempts`: retries on transient transport / 5xx errors
      - `citation_retries`: retries when generation passes wire-level but the
        gate rejects a hallucinated chunk_id. We resample the model rather
        than silently strip the bad citation. If we exhaust the budget, the
        CitationError propagates loudly.
    """
    _log(f"[gen] section: {section_id}")
    system, user = _section_payload(transcript, section_id)
    client = _client()
    last_err: Exception | None = None

    for cite_attempt in range(1, citation_retries + 2):  # +1 for the initial try
        for attempt in range(1, attempts + 1):
            try:
                chunks: list[str] = []
                with client.messages.stream(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                ) as stream:
                    for delta in stream.text_stream:
                        chunks.append(delta)
                markdown = "".join(chunks)
                try:
                    cited = _verify_citations(markdown)
                except CitationError as e:
                    if cite_attempt <= citation_retries:
                        _log(f"[gen]   ! citation gate rejected sample {cite_attempt}; resampling")
                        last_err = e
                        # break to outer loop to resample
                        break
                    raise
                else:
                    _log(f"[gen]   {len(cited)} citation(s), all resolved.")
                    return GeneratedSection(section_id=section_id, markdown=markdown, cited_chunk_ids=cited)
            except APIConnectionError as e:
                last_err = e
                wait = 2 ** attempt
                _log(f"[gen]   ! APIConnectionError on attempt {attempt}/{attempts}; retrying in {wait}s")
                time.sleep(wait)
            except APIStatusError as e:
                if 500 <= e.status_code < 600:
                    last_err = e
                    wait = 2 ** attempt
                    _log(f"[gen]   ! {e.status_code} on attempt {attempt}/{attempts}; retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
        else:
            # Exhausted transport attempts without break — give up.
            break
    assert last_err is not None
    raise last_err


def generate_full_aar(transcript: Transcript) -> FullAAR:
    aar = FullAAR(transcript_title=transcript.title)
    for sid in SECTION_ORDER:
        aar.sections.append(generate_section(transcript, sid))
    return aar


def generate_section_streaming(transcript: Transcript, section_id: str) -> Iterator[dict]:
    """Yields events for SSE-style consumption:
        {"type": "section_start", "section_id": ...}
        {"type": "token", "section_id": ..., "text": "..."}
        {"type": "section_end", "section_id": ..., "cited_chunk_ids": [...]}
    On citation failure:
        {"type": "section_error", "section_id": ..., "error": "...", "raw_markdown": "..."}
    """
    yield {"type": "section_start", "section_id": section_id}
    system, user = _section_payload(transcript, section_id)
    client = _client()
    accumulated = []
    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for delta in stream.text_stream:
                accumulated.append(delta)
                yield {"type": "token", "section_id": section_id, "text": delta}
        markdown = "".join(accumulated)
        cited = _verify_citations(markdown)
        yield {
            "type": "section_end",
            "section_id": section_id,
            "cited_chunk_ids": cited,
            "markdown": markdown,
        }
    except CitationError as e:
        yield {
            "type": "section_error",
            "section_id": section_id,
            "error": str(e),
            "raw_markdown": "".join(accumulated),
        }


def generate_full_aar_streaming(transcript: Transcript) -> Iterator[dict]:
    yield {"type": "aar_start", "title": transcript.title}
    for sid in SECTION_ORDER:
        for ev in generate_section_streaming(transcript, sid):
            yield ev
    yield {"type": "aar_end"}


def _cli():
    if len(sys.argv) < 2:
        print(
            "usage: python -m src.generate <transcript_json> [--full | --section <id>]",
            file=sys.stderr,
        )
        sys.exit(2)
    transcript_path = sys.argv[1]
    transcript = load_transcript(transcript_path)
    args = sys.argv[2:]
    if "--full" in args:
        aar = generate_full_aar(transcript)
        print()
        print(aar.render())
    else:
        section_id = "what_happened"
        if "--section" in args:
            i = args.index("--section")
            section_id = args[i + 1]
        out = generate_section(transcript, section_id)
        print()
        print(out.markdown)


if __name__ == "__main__":
    _cli()
