# NoteCast: plan of record

This builds on the original plan (per-course notebooks → ingest → embed → grounded chat → audio overview). Changes from that draft are marked **[changed]** or **[new]**, with the reason.

---

## 1. Notebooks = per-course folders + one LanceDB table each

```
notebooks/<course-slug>/
  sources/        raw files you drop in (any nesting, e.g. week-03/)
  processed/      extracted text per source (JSON), for debugging and re-chunking
  manifest.json   file → sha256, parser, chunk count, ingested_at
  index.lancedb/  vector + full-text index for this course
  audio/          generated overviews
```

- Switching notebooks means switching the table. As originally planned.
- **[new] Incremental ingest via content hashes.** Re-running `ingest` only processes new or changed files and deletes chunks from removed ones. This matters once you're adding a lecture a week.
- **[new] Week/topic inferred from folder names** (`week-03/`, `wk3`, `topic-graphs/`) with an optional override in `notebooks/<course>/course.yaml`.

## 2. Ingestion

| Format | Parser | Location metadata kept |
|---|---|---|
| PDF | `pymupdf` | page number |
| PPTX | `python-pptx` (slide text **+ speaker notes**) | slide number |
| DOCX | `python-docx` | heading path |
| VTT / SRT | `webvtt-py` / `srt` | start/end timestamp |
| TXT / MD | built-in | heading path |
| **[new]** MP3 / MP4 / M4A | `faster-whisper` → VTT, then as above | timestamp |
| **[new]** Scanned PDFs | OCR fallback (`ocrmypdf`/Tesseract) when a page has no text layer | page |

**Chunking [changed]:** structure-aware, not fixed-size.
- Slides: one chunk per slide (title + body + notes).
- Transcripts: ~60–90 s windows (≈300–400 tokens), split on sentence boundaries, with ~15% overlap. Each chunk keeps `start`/`end` timestamps.
- PDFs/docs: split by heading, then paragraph, targeting ~400 tokens.
- Every chunk gets `{course, source_path, source_type, week, topic, page|slide|t_start,t_end, chunk_id}`.
- **[new] Contextual header.** Prepend a one-line header to the embedded text (`"COMP3000 · Week 3 · Lecture 2 – Graph search · slide 14"`). It noticeably improves retrieval for short or ambiguous chunks and costs nothing.

## 3. Retrieval [changed]

- **LanceDB over Chroma.** LanceDB has built-in full-text search, which makes **hybrid search** (vector + BM25-style keyword) easy. Lecture material is full of exact terms, acronyms, formula names, and code identifiers that pure embeddings miss.
- **Embeddings: `BAAI/bge-small-en-v1.5` instead of `all-MiniLM-L6-v2`.** It's similarly small and fast on CPU and clearly better at retrieval. MiniLM truncates at 256 word-pieces, which silently cuts longer chunks. The embedder is behind an interface, so Voyage or a larger local model is a config change.
- **Optional reranker** (`bge-reranker-base`, local) over the top ~30 hybrid hits → top 8–12 sent to Claude. Toggle it on if eval shows it helps.
- **Filters:** "only week 3", "only the tutorials", "only this file".

## 4. Grounded chat

- Retrieved chunks are sent to Claude as **document blocks with native citations enabled**. Claude's response comes back with structured citations pointing at exact cited text, so we map each one to *file + page/slide/timestamp* reliably instead of parsing `[1]` markers out of prose.
- **Sources-only mode:** system prompt requires answering only from the provided documents, and when they don't contain the answer Claude must say so and give the closest related material it did find.
- **Open mode:** same retrieval, plus Claude's own knowledge and the server-side **web search tool**. The UI labels which parts came from your notes, which came from the web, and which are uncited model knowledge.
- **[new] "Deep" mode (full-context).** Claude now has a **1M-token context window** on current models, so a week's worth of material (or a whole course for many units) can go in **directly**, cached with **prompt caching** so follow-up questions are cheap. This works better than top-k for "compare lecture 2 and lecture 5" or "what are all the assumptions in this unit" questions, where retrieval tends to miss pieces. Default stays RAG, and Deep mode is a toggle.
- Multi-turn: keep chat history. Rewrite follow-ups ("what about the second one?") into standalone queries before retrieval.
- Model IDs are in config. Suggested defaults: **Sonnet 5** (`claude-sonnet-5`, $2/$10 per M tokens) for chat, **Opus 5** (`claude-opus-5`) for audio scripts and Deep mode, **Haiku 4.5** for cheap query rewriting.

## 5. Audio overview [changed: coverage over similarity]

**Top-k retrieval is the wrong tool for summaries.** An overview needs *coverage* of the material, not the chunks most similar to a query. So:

1. **Select scope:** a week, topic, file set, or whole course, plus an optional focus prompt ("focus on exam-relevant stuff").
2. **Gather all chunks in scope.** If it fits (usually true with 1M context), send it all. If not, map-reduce: summarise per source, then write the script from the summaries.
3. **Outline, then script.** Claude writes an outline (key ideas, examples, misconceptions), then a two-host dialogue as **structured JSON** (`[{speaker, text}]`), grounded strictly in the sources. Length target is in minutes (~150 words/min).
4. **TTS:** Kokoro with two distinct voices (Piper fallback). Synthesise line by line, then `pydub` stitches it with short pauses → MP3 (needs `ffmpeg`).
5. **[new] Output a transcript with chapter markers** alongside the MP3, with each chapter linked to its sources.

Formats: `deep-dive` (two hosts, 10–20 min), `brief` (one voice, 3–5 min), `exam-review` (Q&A style).

## 6. [new] Evaluation (small but essential)

A `tests/eval/` set of ~20–30 question/answer/source triples per test course, including **questions whose answer is *not* in the material**. We track:
- retrieval hit rate (was the right chunk in top-k?)
- refusal correctness in sources-only mode
- citation validity (does the cited text actually support the claim?)

This is how we'll know if hybrid search, the reranker, or chunk size changes actually help.

## 7. Interface: to be decided (see questions)

Options: CLI only → Streamlit/Gradio web UI → FastAPI + a proper React front end.

---

## Build phases (each ends with verify → commit → push)

Parallel tracks are in brackets; they run as simultaneous Sonnet builders.

| Phase | Deliverable | Parallel tracks |
|---|---|---|
| **0. Scaffold** | `pyproject` (uv), ruff, pytest, config, `.env.example`, core data models (`Chunk`, `Source`, `Notebook`), and interfaces for parser/embedder/store | single builder |
| **1. Ingestion** | all parsers + chunker + manifest/incremental ingest | [PDF] [PPTX+DOCX] [VTT/SRT/TXT] [chunker+manifest] |
| **2. Index + retrieval** | embedder, LanceDB store, hybrid search, filters | [embedder] [store+hybrid] [eval harness] |
| **3. Grounded chat** | Claude client with citations, sources-only mode, citation → location mapping, CLI chat | [claude client+prompts] [citation mapper] [CLI] |
| **4. Modes** | open mode + web search, Deep (full-context + caching) mode | [web search] [deep mode] |
| **5. Multi-notebook** | create/list/switch/delete courses, scope filters | single builder |
| **6. Audio overview** | scope gather → outline → script JSON → Kokoro TTS → stitch | [script gen] [TTS+stitch] |
| **7. UI** | chosen front end | depends on choice |
| **8. Extras** | whisper transcription, OCR, quiz/flashcards/study guide | parallel per feature |

Phase 1 must be solid before anything else, same as the original plan.
