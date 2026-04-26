"""Post-AAR analysis: timeline markers, finding traces, eval against ground
truth, and doctrine gap detection.

Runs over a generated AAR (either freshly produced or replayed from cache) plus
the original transcript. Emits a structured analysis JSON that the UI consumes
to render the timeline strip, drill panels, eval harness, and gap panel.

Pipeline:
  1. Parse structured findings from key_issues + force_protection markdown.
  2. LLM pass: link each finding to specific transcript entry indexes.
  3. Build per-entry timeline markers (color = SUSTAIN/IMPROVE/observation).
  4. (csis_first_battle only) LLM-as-judge eval vs. published CSIS findings.
  5. LLM pass: doctrine gap detection (contradictions + silences).

The analysis is cached at examples/cache/{demo_id}.analysis.json so it survives
across reloads and replay sessions.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from . import retrieval
from .transcript import Transcript


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_MODEL = "claude-sonnet-4-6"  # cheaper/faster than opus for structured passes
ANALYSIS_MAX_TOKENS = 4000

CITE_FULL = re.compile(r"\[([^\]]+)\]\{\{\s*cite:([^}]+?)\s*\}\s*\}")
CITE_BARE = re.compile(r"\{\{\s*cite:([^}]+?)\s*\}\s*\}")


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    id: str                     # f1, f2, ...
    section_id: str             # key_issues or force_protection
    type: str                   # "SUSTAIN" | "IMPROVE" | "OBSERVATION"
    title: str                  # short headline
    what_supposed: str = ""
    what_actual: str = ""
    why: str = ""
    what_to_do: str = ""
    doctrinal_basis: str = ""
    citation_chunk_ids: list[str] = field(default_factory=list)
    transcript_entry_indexes: list[int] = field(default_factory=list)


@dataclass
class TimelineMarker:
    entry_index: int
    name: str
    summary: str                # one-line synopsis
    classification: str         # "improve" | "sustain" | "observation"
    finding_ids: list[str] = field(default_factory=list)
    citation_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class EvalRow:
    ground_truth_id: str
    ground_truth_text: str
    recovered: bool
    matched_finding_ids: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class EvalReport:
    available: bool                 # only true for csis_first_battle
    recovered: int = 0
    total: int = 0
    novel_finding_ids: list[str] = field(default_factory=list)
    rows: list[EvalRow] = field(default_factory=list)


@dataclass
class GapItem:
    finding_id: str
    kind: str                       # "contradiction" | "silence"
    note: str


@dataclass
class GapReport:
    contradictions: list[GapItem] = field(default_factory=list)
    silences: list[GapItem] = field(default_factory=list)


@dataclass
class AnalysisResult:
    demo_id: str
    transcript_entries: list[dict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    timeline: list[TimelineMarker] = field(default_factory=list)
    eval: EvalReport = field(default_factory=lambda: EvalReport(available=False))
    gaps: GapReport = field(default_factory=GapReport)

    def to_json(self) -> dict[str, Any]:
        return {
            "demo_id": self.demo_id,
            "transcript_entries": self.transcript_entries,
            "findings": [asdict(f) for f in self.findings],
            "timeline": [asdict(m) for m in self.timeline],
            "eval": {
                "available": self.eval.available,
                "recovered": self.eval.recovered,
                "total": self.eval.total,
                "novel_finding_ids": self.eval.novel_finding_ids,
                "rows": [asdict(r) for r in self.eval.rows],
            },
            "gaps": {
                "contradictions": [asdict(g) for g in self.gaps.contradictions],
                "silences": [asdict(g) for g in self.gaps.silences],
            },
        }


# ---------------------------------------------------------------------------
# Markdown -> findings parsing
# ---------------------------------------------------------------------------


def _extract_chunk_ids(text: str) -> list[str]:
    seen: list[str] = []
    for m in CITE_FULL.finditer(text):
        cid = m.group(2).strip()
        if cid not in seen:
            seen.append(cid)
    for m in CITE_BARE.finditer(text):
        cid = m.group(1).strip()
        if cid not in seen:
            seen.append(cid)
    return seen


def _strip_inline_marks(text: str) -> str:
    """Drop bold and inline citation markup so the result is clean prose."""
    out = CITE_FULL.sub(lambda m: m.group(1), text)
    out = CITE_BARE.sub("", out)
    out = out.replace("**", "")
    return out.strip()


_FIELD_KEYS = {
    "what was supposed to happen": "what_supposed",
    "what actually happened": "what_actual",
    "why": "why",
    "what to do about it": "what_to_do",
    "doctrinal basis": "doctrinal_basis",
}


def _classify_title(title: str) -> str:
    upper = title.upper()
    if "(IMPROVE)" in upper:
        return "IMPROVE"
    if "(SUSTAIN)" in upper:
        return "SUSTAIN"
    return ""


def _parse_findings_from_section(section_id: str, markdown: str, start_id: int) -> list[Finding]:
    """Tolerant parser for the SUSTAIN/IMPROVE bullet structure.

    Two formats appear in the corpus:
      A) An explicit `### SUSTAIN` / `### IMPROVE` header before a list of bullets.
      B) Per-bullet classification embedded in the title, e.g.
         `- **Cyber Survivability of Fires Kill Chains (SUSTAIN)**`.

    We accept either. Findings without an explicit class default to OBSERVATION.
    """
    findings: list[Finding] = []
    next_id = start_id
    current_class = ""
    lines = markdown.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # Update current class if we see a header.
        h = re.match(r"^#{2,4}\s+(.*)$", line)
        if h:
            heading = h.group(1).strip().upper()
            if "SUSTAIN" in heading and "IMPROVE" not in heading:
                current_class = "SUSTAIN"
            elif "IMPROVE" in heading and "SUSTAIN" not in heading:
                current_class = "IMPROVE"
            i += 1
            continue

        # Top-level bullet starting a finding: `- **Title.**`
        m = re.match(r"^-\s+\*\*(.+?)\*\*\s*$", line)
        if not m:
            i += 1
            continue
        title_raw = m.group(1).strip().rstrip(".")
        title = _strip_inline_marks(title_raw)

        # Classification: prefer inline tag in title, else section default.
        per_bullet_class = _classify_title(title_raw)
        ftype = per_bullet_class or current_class or "OBSERVATION"
        # Strip the trailing "(SUSTAIN)"/"(IMPROVE)" marker from display title.
        display_title = re.sub(r"\s*\((?:SUSTAIN|IMPROVE)\)\s*$", "", title, flags=re.IGNORECASE).strip()

        # Walk the indented sub-bullets that follow.
        j = i + 1
        fields: dict[str, str] = {}
        while j < len(lines):
            sub = lines[j]
            if re.match(r"^-\s+\*\*", sub):
                # Next top-level bullet starts a new finding.
                break
            if re.match(r"^#{2,4}\s+", sub):
                break
            sub_match = re.match(r"^\s+-\s+\*\*([^*]+?):\*\*\s*(.*)$", sub)
            if sub_match:
                key = sub_match.group(1).strip().lower()
                val_first = sub_match.group(2).strip()
                # Continuation lines are indented further; collect them too.
                k = j + 1
                cont: list[str] = [val_first] if val_first else []
                while k < len(lines):
                    nxt = lines[k]
                    if not nxt.strip():
                        break
                    if re.match(r"^\s+-\s+\*\*", nxt):
                        break
                    if re.match(r"^-\s+\*\*", nxt):
                        break
                    cont.append(nxt.strip())
                    k += 1
                slot = _FIELD_KEYS.get(key)
                if slot:
                    fields[slot] = " ".join(cont).strip()
                j = k
            else:
                j += 1

        full_blob = " ".join(fields.values()) + " " + title_raw
        chunk_ids = _extract_chunk_ids(full_blob)

        if display_title and (fields or per_bullet_class or current_class):
            findings.append(Finding(
                id=f"f{next_id}",
                section_id=section_id,
                type=ftype,
                title=display_title,
                what_supposed=_strip_inline_marks(fields.get("what_supposed", "")),
                what_actual=_strip_inline_marks(fields.get("what_actual", "")),
                why=_strip_inline_marks(fields.get("why", "")),
                what_to_do=_strip_inline_marks(fields.get("what_to_do", "")),
                doctrinal_basis=_strip_inline_marks(fields.get("doctrinal_basis", "")),
                citation_chunk_ids=chunk_ids,
            ))
            next_id += 1
        i = j

    return findings


def parse_findings(section_markdown: dict[str, str]) -> list[Finding]:
    """Parse findings from key_issues and force_protection sections.

    section_markdown maps section_id -> markdown string (post-citation-gate).
    """
    findings: list[Finding] = []
    next_id = 1
    for sid in ("key_issues", "force_protection"):
        md = section_markdown.get(sid, "")
        if not md:
            continue
        section_findings = _parse_findings_from_section(sid, md, next_id)
        findings.extend(section_findings)
        next_id += len(section_findings)
    return findings


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _client() -> Anthropic:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return Anthropic()


def _llm_json(system: str, user: str, *, model: str = ANALYSIS_MODEL) -> Any:
    """Call the LLM and parse a JSON object out of its response.

    Tolerant of fenced code blocks. If parsing fails, returns None — callers
    decide how to degrade. We do not crash the analysis pipeline on a single
    bad LLM call; partial analysis is still useful.
    """
    client = _client()
    resp = client.messages.create(
        model=model,
        max_tokens=ANALYSIS_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    text = text.strip()
    # Strip ```json fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Find the first { and last } if there's noise around the JSON.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        # Maybe a top-level array.
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end < start:
            return None
    blob = text[start:end + 1]
    try:
        return json.loads(blob)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 2: link findings to transcript entries
# ---------------------------------------------------------------------------


_LINK_SYSTEM = (
    "You are an analyst mapping after-action findings to the wargame transcript "
    "moments that ground them. Be conservative: only link a finding to an entry "
    "when the entry text directly evidences the finding. Output JSON only."
)


def _link_findings_to_transcript(transcript: Transcript, findings: list[Finding]) -> dict[str, list[int]]:
    if not findings or not transcript.entries:
        return {}

    entry_blocks = []
    for i, e in enumerate(transcript.entries):
        snippet = e.text.strip().replace("\n", " ")
        entry_blocks.append(f"[{i}] {e.name}: {snippet}")
    transcript_block = "\n".join(entry_blocks)

    finding_blocks = []
    for f in findings:
        summary_parts = [f"{f.type}: {f.title}"]
        if f.what_actual:
            summary_parts.append(f"What happened: {f.what_actual}")
        elif f.why:
            summary_parts.append(f"Why: {f.why}")
        finding_blocks.append(f"{f.id} | " + " | ".join(summary_parts))
    findings_block = "\n".join(finding_blocks)

    user = (
        "TRANSCRIPT ENTRIES (numbered):\n"
        f"{transcript_block}\n\n"
        "FINDINGS:\n"
        f"{findings_block}\n\n"
        "For each finding, list the indexes of the transcript entries that "
        "directly evidence it. Each finding should map to 1-3 entries; skip a "
        "finding if no entry evidences it. Use only the integer indexes shown.\n\n"
        'Return JSON of the form {"links": [{"finding_id": "f1", "entry_indexes": [3, 4]}]}.'
    )
    parsed = _llm_json(_LINK_SYSTEM, user)
    out: dict[str, list[int]] = {}
    if not parsed or not isinstance(parsed, dict):
        return out
    for row in parsed.get("links", []):
        fid = row.get("finding_id")
        idxs = row.get("entry_indexes", [])
        if not isinstance(fid, str) or not isinstance(idxs, list):
            continue
        clean = []
        for x in idxs:
            try:
                xi = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= xi < len(transcript.entries) and xi not in clean:
                clean.append(xi)
        if clean:
            out[fid] = clean
    return out


# ---------------------------------------------------------------------------
# Step 3: build timeline markers
# ---------------------------------------------------------------------------


def _entry_summary(entry_text: str, max_chars: int = 140) -> str:
    """First sentence (or first line) trimmed to max_chars."""
    text = entry_text.strip().replace("\n", " ")
    # Pull the first sentence-ish unit.
    m = re.match(r"^(.{20,}?[.!?])(\s|$)", text)
    head = m.group(1) if m else text
    if len(head) > max_chars:
        head = head[: max_chars - 1].rstrip() + "…"
    return head


def _build_timeline(transcript: Transcript, findings: list[Finding]) -> list[TimelineMarker]:
    by_entry: dict[int, list[Finding]] = {}
    for f in findings:
        for idx in f.transcript_entry_indexes:
            by_entry.setdefault(idx, []).append(f)

    markers: list[TimelineMarker] = []
    for i, e in enumerate(transcript.entries):
        attached = by_entry.get(i, [])
        if any(f.type == "IMPROVE" for f in attached):
            cls = "improve"
        elif any(f.type == "SUSTAIN" for f in attached):
            cls = "sustain"
        else:
            cls = "observation"
        cite_ids: list[str] = []
        for f in attached:
            for cid in f.citation_chunk_ids:
                if cid not in cite_ids:
                    cite_ids.append(cid)
        markers.append(TimelineMarker(
            entry_index=i,
            name=e.name,
            summary=_entry_summary(e.text),
            classification=cls,
            finding_ids=[f.id for f in attached],
            citation_chunk_ids=cite_ids,
        ))
    return markers


# ---------------------------------------------------------------------------
# Step 4: eval against published CSIS ground truth
# ---------------------------------------------------------------------------


# Hardcoded ground-truth findings from the CSIS "First Battle of the Next War"
# (Cancian, Cancian, Heginbotham, January 2023). These are the published key
# findings in the executive summary, paraphrased to a stable form so an LLM
# judge can match against generator output.
CSIS_GROUND_TRUTH = [
    {
        "id": "csis1",
        "text": "The United States and its allies can defeat a Chinese amphibious invasion of Taiwan in the base case, but at very high cost.",
    },
    {
        "id": "csis2",
        "text": "The United States loses about two carriers and 10-20 large surface combatants in most scenarios.",
    },
    {
        "id": "csis3",
        "text": "Taiwan must hold the line: ground forces must blunt and contain any Chinese lodgment until US/allied fires can be brought to bear.",
    },
    {
        "id": "csis4",
        "text": "The United States rapidly exhausts its inventory of long-range precision-guided munitions, especially anti-ship missiles like LRASM and JASSM-ER, within the first week of high-intensity combat.",
    },
    {
        "id": "csis5",
        "text": "Submarines (SSNs) are highly effective against Chinese amphibious shipping and provide the decisive maritime fires against PLAN amphibs.",
    },
    {
        "id": "csis6",
        "text": "US bombers operating from outside the threat envelope, especially from Guam and CONUS with stand-off munitions, are key contributors to anti-ship strikes.",
    },
    {
        "id": "csis7",
        "text": "Japan is critical: access to Japanese bases and Japanese forces themselves materially affects the outcome; without Japanese basing the US cannot effectively defend Taiwan.",
    },
    {
        "id": "csis8",
        "text": "Forward-deployed US air assets at Kadena and other regional bases suffer heavy losses to PLA missile strikes; hardened shelters and dispersal are necessary.",
    },
    {
        "id": "csis9",
        "text": "The defense industrial base is a binding constraint; munitions production cannot keep pace with wartime expenditure rates and must be expanded in peacetime.",
    },
]


_EVAL_SYSTEM = (
    "You are an evaluator comparing generated AAR findings against published "
    "CSIS ground-truth findings from 'The First Battle of the Next War'. For "
    "each ground-truth finding, decide whether any generated finding "
    "substantively recovers it (same operational claim, even if different "
    "wording). Be strict: surface-level overlap is not recovery. Output JSON only."
)


def _eval_against_csis(findings: list[Finding]) -> EvalReport:
    if not findings:
        return EvalReport(available=True, total=len(CSIS_GROUND_TRUTH))

    findings_block = "\n".join(
        f"{f.id} [{f.type}] {f.title}: {f.what_actual or f.why or ''}"[:400]
        for f in findings
    )
    truth_block = "\n".join(f"{g['id']}: {g['text']}" for g in CSIS_GROUND_TRUTH)

    user = (
        "GROUND-TRUTH FINDINGS (CSIS):\n"
        f"{truth_block}\n\n"
        "GENERATED FINDINGS:\n"
        f"{findings_block}\n\n"
        "For each ground-truth finding, decide whether the generated set "
        "recovers it. Recovery requires substantive operational match, not "
        "topical overlap. Multiple generated findings may map to the same "
        "ground truth.\n\n"
        "Return JSON: "
        '{"rows": [{"ground_truth_id": "csis1", "recovered": true, '
        '"matched_finding_ids": ["f3"], "rationale": "..."}], '
        '"novel_finding_ids": ["f8"]}. '
        "novel_finding_ids = generated finding ids that do not match any ground truth."
    )
    parsed = _llm_json(_EVAL_SYSTEM, user)
    if not parsed or not isinstance(parsed, dict):
        return EvalReport(
            available=True,
            total=len(CSIS_GROUND_TRUTH),
            rows=[EvalRow(g["id"], g["text"], False) for g in CSIS_GROUND_TRUTH],
        )

    by_id = {r.get("ground_truth_id"): r for r in parsed.get("rows", []) if isinstance(r, dict)}
    rows: list[EvalRow] = []
    recovered_count = 0
    for g in CSIS_GROUND_TRUTH:
        r = by_id.get(g["id"], {})
        is_rec = bool(r.get("recovered"))
        matched = [m for m in r.get("matched_finding_ids", []) if isinstance(m, str)]
        rationale = r.get("rationale", "") or ""
        if is_rec:
            recovered_count += 1
        rows.append(EvalRow(
            ground_truth_id=g["id"],
            ground_truth_text=g["text"],
            recovered=is_rec,
            matched_finding_ids=matched,
            rationale=rationale,
        ))

    novel_ids = [n for n in parsed.get("novel_finding_ids", []) if isinstance(n, str)]
    return EvalReport(
        available=True,
        recovered=recovered_count,
        total=len(CSIS_GROUND_TRUTH),
        novel_finding_ids=novel_ids,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Step 5: doctrine gap detection
# ---------------------------------------------------------------------------


_GAP_SYSTEM = (
    "You are a doctrine analyst reviewing AAR findings for gaps. Identify two "
    "kinds of gaps: (1) CONTRADICTIONS — findings where the wargame outcome "
    "diverges from what the cited or expected doctrine would predict, and "
    "(2) SILENCES — findings where doctrine offers no clear guidance and the "
    "topic should be flagged for J7 research. Be specific and cite the "
    "finding_id. Output JSON only."
)


def _find_doctrine_gaps(findings: list[Finding]) -> GapReport:
    if not findings:
        return GapReport()

    blocks = []
    for f in findings:
        cite_str = ", ".join(f.citation_chunk_ids) if f.citation_chunk_ids else "(no citations)"
        blocks.append(
            f"{f.id} [{f.type}] {f.title}\n"
            f"  what actually happened: {f.what_actual}\n"
            f"  why: {f.why}\n"
            f"  doctrinal basis: {f.doctrinal_basis}\n"
            f"  cited chunks: {cite_str}"
        )
    findings_block = "\n\n".join(blocks)

    user = (
        "FINDINGS:\n"
        f"{findings_block}\n\n"
        "Identify contradictions and silences. A contradiction is where the "
        "actual outcome diverges from what cited (or standard) doctrine would "
        "predict — flag this as a doctrine-validity question. A silence is a "
        "finding where the corpus produced no clearly relevant doctrinal "
        "passage, or the cited basis is thin — flag this as a J7 research "
        "question.\n\n"
        "Return JSON: "
        '{"contradictions": [{"finding_id": "f5", "note": "..."}], '
        '"silences": [{"finding_id": "f8", "note": "..."}]}. '
        "Limit to at most 4 items per category. Skip categories with no examples."
    )
    parsed = _llm_json(_GAP_SYSTEM, user)
    if not parsed or not isinstance(parsed, dict):
        return GapReport()

    valid_ids = {f.id for f in findings}

    def _coerce(items: list[dict], kind: str) -> list[GapItem]:
        out: list[GapItem] = []
        for it in items or []:
            if not isinstance(it, dict):
                continue
            fid = it.get("finding_id")
            note = it.get("note", "")
            if fid in valid_ids and isinstance(note, str) and note:
                out.append(GapItem(finding_id=fid, kind=kind, note=note))
        return out

    return GapReport(
        contradictions=_coerce(parsed.get("contradictions", []), "contradiction"),
        silences=_coerce(parsed.get("silences", []), "silence"),
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def section_markdowns_from_events(events: list[dict]) -> dict[str, str]:
    """Pull successful section markdown out of a cached event stream."""
    out: dict[str, str] = {}
    for ev in events:
        if ev.get("type") == "section_end":
            sid = ev.get("section_id")
            md = ev.get("markdown")
            if sid and md:
                out[sid] = md
    return out


def analyze(
    demo_id: str,
    transcript: Transcript,
    section_markdown: dict[str, str],
    *,
    run_eval: bool = True,
    run_gaps: bool = True,
) -> AnalysisResult:
    """Full analysis pipeline. LLM steps degrade gracefully — a failed LLM call
    leaves that section empty rather than blowing up the whole analysis."""
    findings = parse_findings(section_markdown)

    # LLM step: link findings to transcript entries.
    try:
        links = _link_findings_to_transcript(transcript, findings)
    except Exception:
        links = {}
    for f in findings:
        f.transcript_entry_indexes = links.get(f.id, [])

    timeline = _build_timeline(transcript, findings)

    if run_eval and demo_id == "csis_first_battle":
        try:
            eval_report = _eval_against_csis(findings)
        except Exception:
            eval_report = EvalReport(
                available=True,
                total=len(CSIS_GROUND_TRUTH),
                rows=[EvalRow(g["id"], g["text"], False) for g in CSIS_GROUND_TRUTH],
            )
    else:
        eval_report = EvalReport(available=False)

    if run_gaps and findings:
        try:
            gap_report = _find_doctrine_gaps(findings)
        except Exception:
            gap_report = GapReport()
    else:
        gap_report = GapReport()

    return AnalysisResult(
        demo_id=demo_id,
        transcript_entries=[
            {"index": i, "name": e.name, "text": e.text}
            for i, e in enumerate(transcript.entries)
        ],
        findings=findings,
        timeline=timeline,
        eval=eval_report,
        gaps=gap_report,
    )
