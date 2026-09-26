# NoteCast

A self-hosted, Claude-powered take on NotebookLM for course material.

Drop your lectures, workshops, tutorials, slides, and transcripts into a per-course notebook. Then:

- **Ask questions** and get answers grounded in your material, with citations back to the exact file, slide, page, or timestamp ("at 14:32 the lecturer said…").
- **Switch modes**: *Sources only* (answers strictly from your notes, and it says so when the answer isn't there) or *Open* (Claude can also use its own knowledge plus web search, clearly marked as not coming from your sources).
- **Generate audio overviews**: a two-host podcast-style episode or a single-voice recap for a week, a topic, or the whole course, rendered locally with open-weight TTS.

There's no chat attachment limit because your material lives in a local index. Claude only sees the passages relevant to each question, or a whole week's content when that's what the task needs.

> **Status:** Phases 0–7 done and verified (ingest, search, grounded chat in
> sources/open/deep modes, multi-notebook, audio overviews, web UI). Next: a
> real end-to-end run with an API key. See [`docs/LEARN.md`](docs/LEARN.md) and
> [`docs/PLAN.md`](docs/PLAN.md).

> **This branch:** adds **AgentAUS** (Trellis Data's sovereign, OpenAI-compatible
> model) as an alternative LLM provider, and it's the default here
> (`NOTECAST_PROVIDER=agentaus`). Claude is still available
> (`NOTECAST_PROVIDER=anthropic`). See [`docs/AGENTAUS.md`](docs/AGENTAUS.md)
> for setup and the differences from Claude.

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

## Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+, managed with `uv` |
| Parsing | `pymupdf` (PDF), `python-pptx`, `python-docx`, `webvtt-py` / `srt` |
| Vector store | LanceDB (embedded, built-in full-text search for hybrid retrieval) |
| Embeddings | `fastembed` (ONNX) running `BAAI/bge-small-en-v1.5` locally — no GPU/PyTorch needed |
| LLM | AgentAUS (default on this branch) or Claude — see [`docs/AGENTAUS.md`](docs/AGENTAUS.md). Claude uses the official `anthropic` SDK, with native citations, prompt caching, and web search. AgentAUS uses the `openai` SDK against its OpenAI-compatible API, with those features emulated (see the docs page for the differences) |
| TTS | Kokoro via `kokoro-onnx`, runs locally (free); MP3 via `lameenc` |
| UI | Streamlit (local web UI) + thin CLI |

## Layout

```
notecast/                   # Python package
  ingest/                   # parsers + chunker
  index/                    # embeddings + LanceDB
  chat/                     # retrieval + Claude calls
  audio/                    # script gen + TTS
notebooks/<course>/         # your material — git-ignored, never uploaded
  sources/                  # raw files you drop in (any nesting, e.g. week-01/)
  processed/                # extracted text + manifest, for debugging
  index.lancedb/            # this course's search index
  audio/                    # generated overviews
tests/
docs/PLAN.md
```

## Quickstart

New to command-line dev tools, or on a fresh Windows machine? Start with the
full step-by-step guide: **[`docs/SETUP-WINDOWS.md`](docs/SETUP-WINDOWS.md)**.
Want to understand *why* NoteCast is built this way (chunks, embeddings, RAG,
citations)? See **[`docs/LEARN.md`](docs/LEARN.md)**, which grows alongside
the code, one section per build phase.

If you already have `uv` installed:

```bash
uv sync
cp .env.example .env                 # then add your AgentAUS or Anthropic key (see docs/AGENTAUS.md / docs/SETUP-WINDOWS.md)
uv run notecast provider-check       # confirm the key, base URL and model ID work

uv run notecast notebooks create comp3000
# copy your files into notebooks/comp3000/sources/, e.g. sources/week-01/

uv run notecast ingest comp3000      # parse + index (first run downloads the embedding model)
uv run notecast ask comp3000 "What is TF-IDF?"
uv run notecast chat comp3000 --mode deep      # whole-course context, shows a cost estimate first
uv run notecast audio comp3000 --week 1        # generate a two-host audio overview

uv run notecast ui        # local web UI: chat + audio, same core as the CLI
```

For the full day-to-day guide (every command, all three chat modes,
what the audio pipeline does, a cost cheat-sheet, and troubleshooting), see
**[`docs/USAGE.md`](docs/USAGE.md)**.

## Privacy

Course material, indexes, and generated audio live under `notebooks/` and are
**git-ignored — they are never committed**, because this repository is
**public** and your course material is copyrighted. Nothing in that folder
ever leaves your machine except the specific passages a request needs, sent
to whichever LLM provider is configured (AgentAUS or Claude) to generate an
answer or a script. Embeddings and text-to-speech always run locally, on
either provider.
