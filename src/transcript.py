"""Snow Globe `History` JSON parser.

Snow Globe (`llm_snowglobe.snowglobe.History`) stores entries as a list of
`{"name": str, "text": str}` dicts. Our on-disk format matches that, plus an
optional top-level `meta` block for scenario context the AAR generator can use.

Schema (input JSON):
    {
      "meta": {
        "title": "...",
        "scenario": "...",
        "objectives": ["..."],
        "source": "...",         // optional: cite the source report
        "force_lay": "..."       // optional
      },
      "entries": [
        {"name": "Narrator",   "text": "..."},
        {"name": "Blue Cell",  "text": "..."},
        {"name": "Red Cell",   "text": "..."},
        {"name": "Adjudicator","text": "..."}
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Entry:
    name: str
    text: str


@dataclass
class Transcript:
    title: str = ""
    scenario: str = ""
    objectives: list[str] = field(default_factory=list)
    source: str = ""
    force_lay: str = ""
    entries: list[Entry] = field(default_factory=list)

    def render_for_prompt(self) -> str:
        """Plain-text rendering for inclusion in LLM prompts. Each entry gets
        the speaker's name as a header; this is the closest analog to how
        Snow Globe formats history for downstream prompts (see History.str)."""
        chunks: list[str] = []
        if self.scenario:
            chunks.append(f"SCENARIO: {self.scenario}")
        if self.objectives:
            chunks.append("OBJECTIVES:\n- " + "\n- ".join(self.objectives))
        if self.force_lay:
            chunks.append(f"FORCE LAYDOWN: {self.force_lay}")
        if chunks:
            chunks.append("---")
        for e in self.entries:
            chunks.append(f"[{e.name}]\n{e.text}")
        return "\n\n".join(chunks)


def load(path: str | Path) -> Transcript:
    p = Path(path)
    raw = json.loads(p.read_text())
    meta = raw.get("meta", {})
    entries = [Entry(name=e["name"], text=e["text"]) for e in raw.get("entries", [])]
    return Transcript(
        title=meta.get("title", p.stem),
        scenario=meta.get("scenario", ""),
        objectives=list(meta.get("objectives", [])),
        source=meta.get("source", ""),
        force_lay=meta.get("force_lay", ""),
        entries=entries,
    )
