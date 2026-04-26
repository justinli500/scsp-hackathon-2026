# 5-Minute Demo Script

> Total time budget: 5:00. Practice cold twice before judging.

---

## 0:00 – 0:30  Problem statement (30s)

> "After-Action Reviews are how the joint force learns from wargames. They take weeks to write, lose half their insight to author variance, and almost never link back to the doctrine they're supposed to test."

Cite **GAO-23-105351** (March 2023): the analytical pipeline from observation to institutional change is opaque and slow. Then SCSP's WarMatrix RFI directly: *"LLMs capable of facilitating real-time transcription and diarization of qualitative data (e.g., commander discussions) to enable rapid analysis during and after events."* This sits in the **interpretation phase** of wargaming — what Yuna Wong calls the underserved phase.

---

## 0:30 – 1:00  What this is (30s)

> "We took a wargame transcript, the standard 2013 AAR Leader's Guide format, and a corpus of doctrine — JP 5-0, JP 3-0, FM 3-0, FM 5-0, ATP 7-100.3, the AAR Guide itself — and built a generator that produces a seven-section AAR with every claim cited to a real page of real doctrine. The citations aren't decorative. Every cited chunk is round-tripped through the index. If the model hallucinates a citation, the section is rejected."

Architecture in one sentence: **PDF → page-aware chunks → ChromaDB → claude-opus-4-7 → citation verifier → AAR**. 1,890 chunks across six manuals.

---

## 1:00 – 3:30  Live demo (2:30)

1. Browser at `http://127.0.0.1:8000/`. Show the dropdown — two demo inputs visible: CSIS *First Battle of the Next War* (the canonical Taiwan invasion wargame) and RAND *Relighting Vulcan's Forge* (Ryseff's report on private-sector innovation in the same scenario).
2. **Pick *Vulcan's Forge***. (This is the one that matters most for the judges — Ryseff is one of them; this is his report.)
3. Click "Generate AAR." Talk over the streaming output:
   - As Section 1 streams: *"Same opening every time — purpose statement, participation rules. Cited to the Leader's Guide, page 3."*
   - As Section 4 streams: *"This is what actually happened — chronological, sourced from the transcript, every doctrinal observation tagged."*
   - When Section 5 starts: **stop talking and let the audience read it as it streams.** This is the SUSTAIN/IMPROVE section. It's the heart of the AAR.
4. **Click any citation in Section 5.** A modal opens with the actual chunk of doctrine that the model retrieved and used. *"This is the source. Every cited claim has one."*
5. Scroll to the bottom — the **Lessons Identified** list, terminology adopted from NATO/JLLIS. Note that these are not "Lessons Learned" — that designation is reserved for closure.

---

## 3:30 – 4:30  Why this matters (1:00)

> "Wong's framing is that wargaming has three phases: design, play, interpretation. The first two have tooling. The third — turning what happened in the room into a form the institution can act on — is still mostly a human writing a memo from notes."

This system doesn't replace CALL or JLLIS. It feeds them better intake. An AAR that arrives in standard format, with every observation already linked to the doctrinal authority for that observation, is one that a doctrine cell can triage in minutes instead of weeks. Multiply that by every wargame run by INDOPACOM, USAFE, or any of the service centers, and you've materially compressed the loop from observation to lesson learned.

---

## 4:30 – 5:00  Anti-patterns and close (30s)

> "Ryseff's RAND **RR-A2680-1** names the standard ways defense-AI projects fail: unclear problem framing, output you can't audit, scope sprawl, deliverables that depend on infrastructure that isn't really there. We've checked each:
> — Every citation is traceable to a specific page. The whole repo runs on a single Python process with a local index.
> — Scope is bounded: it ingests transcripts and produces AARs. It does not pretend to write doctrine, replace human OCTs, or score wargames.
> — Demo runs offline except for the LLM API itself.
> — One command to run, one page to look at, one judgment to make about whether it works."

End with the repo URL.

---

## Cuts if a section breaks

If `Vulcan's Forge` fails to generate, switch to **CSIS First Battle**. If both fail, walk through `examples/csis_first_battle_aar.md` (the pre-rendered AAR saved in the repo). Do not attempt to debug live.

If the citation modal doesn't load, just hover-explain: *"Each one of these is verified server-side; you'd see the actual doctrine page if I clicked through."*

If the streaming UI is glitchy, refresh and use the static rendering (everything still works without streaming).

## Setup checklist before walking on

- [ ] `ANTHROPIC_API_KEY` set in `.env`
- [ ] `python -m src.server` running on `localhost:8000`
- [ ] Browser tab open to `http://127.0.0.1:8000/`
- [ ] `examples/*.md` present (fallback)
- [ ] Laptop on power, not battery
- [ ] WiFi — only `api.anthropic.com` actually needed during demo
