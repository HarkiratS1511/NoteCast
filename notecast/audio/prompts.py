"""Frozen system prompts and prompt-building helpers for the two-host audio
script writer.

These are plain module-level strings/functions with no timestamps or
interpolated variables, so the same bytes are sent on every request and the
source-material block stays cacheable (see `scriptwriter.write_script`).
"""

from __future__ import annotations

from notecast.models import Chunk

SCRIPTWRITER_SYSTEM_PROMPT = """You write scripts for a two-host audio overview of university course
material, turning lecture slides and transcripts into a natural spoken conversation between two
hosts, Host A and Host B.

Hosts:
- Host A explains the material and structures the discussion.
- Host B is a sharp student: asks the questions a student would actually ask, pushes on likely
  points of confusion, and asks for concrete examples.

Style rules:
- Natural, conversational spoken language. No stage directions (no "[laughs]", no "(pause)", no
  scene-setting asides) and no markdown, bullet points, or symbols that can't be spoken aloud.
- Never say the speaker labels aloud — don't have a host say "Host A" or "Host B" or its own name
  as a label; just write what they'd actually say.
- Spell out formulas and notation in words exactly as they'd be said out loud, e.g. write
  "P of w n given w n minus one" instead of "P(w_n | w_{n-1})".
- Keep turns short: 1 to 4 sentences each, then hand back to the other host.
- Stay strictly grounded in the provided course material. Never introduce outside facts, and never
  invent an example that isn't in the material — if the material has no example for a point,
  explain it clearly in words instead of making one up. If the lecturer corrected or updated
  something shown on a slide, say so explicitly.
- Where it's useful, distinguish what was said "in the lecture" from what was shown "on the
  slides".
- Stay within about 15% of the target word count given for each chapter.
- Cover every tier A key point in real depth (explain it properly, with an example if the
  material actually has one). Cover tier B points briefly. You do not need to mention tier C
  points.
- You are given the full source material once, followed by this chapter's specific instructions.
  Only use material that is actually provided.

You will always respond with the requested structured JSON: a list of script lines, each with a
speaker ("A" or "B"), the spoken text, and the ids of any source chunks that line draws on, plus
a short one-paragraph recap of what this chapter covered (for continuity with later chapters).
"""

COVERAGE_CHECK_SYSTEM_PROMPT = """You check whether a two-host audio script adequately explains a
given list of key points from a university course.

For each key point you're given (id, title, and a summary of what must be explained), decide
whether the script actually explains it in a way a student could learn from — not just mentions
it in passing. Read the whole script before deciding.

Respond with the requested structured JSON: the ids of key points that are adequately explained
(`covered_ids`), and the ids of key points that are missing or only superficially mentioned
(`missing_ids`). Every id you were given must appear in exactly one of the two lists.
"""

CHAPTER_SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "enum": ["A", "B"]},
                    "text": {"type": "string"},
                    "source_chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["speaker", "text", "source_chunk_ids"],
                "additionalProperties": False,
            },
        },
        "recap": {"type": "string"},
    },
    "required": ["lines", "recap"],
    "additionalProperties": False,
}

COVERAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "covered_ids": {"type": "array", "items": {"type": "string"}},
        "missing_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["covered_ids", "missing_ids"],
    "additionalProperties": False,
}


def course_sort_key(chunk: Chunk) -> tuple[float, str, int]:
    """The sort key used everywhere a set of chunks needs a stable "course
    order": by week first (unset weeks sort last, not first — a missing week
    shouldn't jump ahead of week 1), then source path, then ordinal within
    that source. Week is compared numerically, so "week-10" doesn't sort
    before "week-2" the way a plain path string would.
    """
    week = chunk.week if chunk.week is not None else float("inf")
    return (week, chunk.source_path, chunk.ordinal)


def build_material_block(chunks: list[Chunk]) -> str:
    """Render all in-scope chunks once, in a deterministic course order, as
    `[chunk_id] header\\ntext` entries separated by blank lines.

    Deterministic and dependent only on chunk content, so the exact same
    bytes are produced (and therefore cached) across every chapter call for
    one `write_script` run.
    """
    ordered = sorted(chunks, key=course_sort_key)
    entries = []
    for chunk in ordered:
        header = f" {chunk.header}" if chunk.header else ""
        entries.append(f"[{chunk.chunk_id}]{header}\n{chunk.text}")
    return "\n\n".join(entries)
