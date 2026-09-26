# NoteCast

A self-hosted, Claude-powered take on NotebookLM for course material.

Drop your lectures, workshops, tutorials, slides, and transcripts into a per-course notebook. Then:

- **Ask questions** and get answers grounded in your material, with citations back to the exact file, slide, page, or timestamp ("at 14:32 the lecturer said…").
- **Switch modes**: *Sources only* (answers strictly from your notes, and it says so when the answer isn't there) or *Open* (Claude can also use its own knowledge plus web search, clearly marked as not coming from your sources).
- **Generate audio overviews**: a two-host podcast-style episode or a single-voice recap for a week, a topic, or the whole course, rendered locally with open-weight TTS.

There's no chat attachment limit because your material lives in a local index. Claude only sees the passages relevant to each question, or a whole week's content when that's what the task needs.

> **Status:** planning. See [`docs/PLAN.md`](docs/PLAN.md) for the architecture, build phases, and open decisions.

## How it works (short version)

```
sources/ ──► parse ──► chunk + metadata ──► embed ──► LanceDB (one table per course)
                                                            │
question ──► hybrid search (vector + keyword) ──► rerank ──►┤
                                                            ▼
                              Claude API (citations on, optional web search)
                                                            │
                                   answer + clickable source citations
```

Audio overviews use the same grounding, then script ► Kokoro TTS (two voices) ► stitched MP3.

## Planned stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+, managed with `uv` |
| Parsing | `pymupdf` (PDF), `python-pptx`, `python-docx`, `webvtt-py` / `srt` |
| Vector store | LanceDB (embedded, built-in full-text search for hybrid retrieval) |
| Embeddings | Local `BAAI/bge-small-en-v1.5` by default, swappable (Voyage optional) |
| LLM | Claude API via the official `anthropic` SDK, with native citations, prompt caching, and web search |
| TTS | Kokoro (default), Piper (fallback); `pydub` + `ffmpeg` for stitching |
| UI | Streamlit (local web UI) + thin CLI |

## Layout (planned)

```
notecast/            # Python package
  ingest/            # parsers + chunker
  index/             # embeddings + LanceDB
  chat/              # retrieval + Claude calls
  audio/             # script gen + TTS
notebooks/<course>/  # your material (git-ignored)
  sources/           # raw files you drop in
  processed/         # extracted text + manifest
tests/
docs/PLAN.md
```

## Setup

Coming once Phase 1 lands. It will look roughly like:

```bash
uv sync
cp .env.example .env        # add ANTHROPIC_API_KEY
notecast add-course comp3000
notecast ingest comp3000
notecast chat comp3000
```

## Privacy

Course material, indexes, and generated audio are git-ignored and stay on your machine. Only the retrieved passages for each request are sent to the Claude API.
