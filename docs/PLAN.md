# NoteCast: plan of record

This builds on the original plan (per-course notebooks → ingest → embed → grounded chat → audio overview). Changes from that draft are marked **[changed]** or **[new]**, with the reason.

## Decisions (2026-09-26)

| Topic | Decision |
|---|---|
| Interface | **Streamlit local web UI** on top of a Python core with a thin CLI for ingest/scripting. The core is UI-agnostic, so a React front end can be added later. |
| Sources | **PDF, PPTX, DOCX, VTT/SRT/TXT/MD.** Whisper transcription and OCR are dropped from scope (they can go back into Phase 8 later). |
| Hardware | **Windows/Linux, CPU by default.** [changed] Embeddings and reranking run via `fastembed` (ONNX runtime) on CPU — no PyTorch/CUDA install needed, and it's fast enough: ~13 passages/sec on a laptop-class CPU, so a whole course (~60–80k tokens) indexes in ~15–20 s. GPU is optional later, if a bigger local model is ever worth it. TTS (Phase 6, see §5) is planned around the same CPU-first, no-PyTorch approach. `ffmpeg` is a documented prerequisite for audio only. |
| API budget | **Cheapest.** Chat defaults to Sonnet 5, and query rewriting and other helper calls use Haiku 4.5. Audio scripts use Sonnet 5 too. Opus isn't used by default and is a config switch. Deep mode is opt-in and shows an estimated token cost before sending. Prompt caching is on wherever the prefix repeats. |
| OS / experience | **Windows, fresh setup, learning as we go.** Phase 0 ships `docs/SETUP-WINDOWS.md` (step-by-step installs for uv, Python, ffmpeg, CUDA PyTorch) and `docs/LEARN.md`, a plain-English explainer that grows each phase (what embeddings are, why hybrid search, how citations work). Paths use `pathlib` everywhere and nothing assumes a Unix shell. |
| Course material | **Never in git; the repo is public and the material is copyrighted.** Material lives in `notebooks/<course>/sources/` (git-ignored). Tests use small synthetic fixtures written for this repo. The real-material eval set lives next to the material, at `notebooks/<course>/eval.yaml`, also git-ignored. |
| Audio | **Two hosts. Length adapts to the material, up to 45 min. Must cover the important points** from both slides and transcripts (see §5). |
| Branching | Work directly on `main`. |
| **[new]** LLM provider | **AgentAUS is the default provider on this branch** (`NOTECAST_PROVIDER=agentaus`), talking to Trellis Data's sovereign, OpenAI-compatible API via an adapter (`notecast/providers/openai_compat.py`). Claude remains fully supported (`NOTECAST_PROVIDER=anthropic`). See below and [`docs/AGENTAUS.md`](AGENTAUS.md). |

**First real course:** COMP4650/6490 Document Analysis, weeks 1–3 slide PDFs (61/94/60 pages, ≈65k characters ≈ 16k tokens total). Findings:
- These are PowerPoint-exported PDFs, so **one PDF page = one slide**, and the PDF parser should chunk per page.
- Every page has a boilerplate footer ("ANU SCHOOL OF COMPUTING | DOCUMENT ANALYSIS | 12"). The parser must **strip repeated headers/footers** (lines recurring on >50% of pages) but keep the page number as metadata.
- A full course at this density is roughly 60–80k tokens. That **easily fits in context**, so Deep mode and full-coverage audio scripts are cheap (~$0.15 uncached with Sonnet 5, ~10× less on cache hits). Retrieval is still the default for chat, since it's cheaper per question and gives tighter citations.

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

**Chunking [changed]:** structure-aware, not fixed-size.
- Slides: one chunk per slide (title + body + notes).
- Transcripts: ~60–90 s windows (≈300–400 tokens), split on sentence boundaries, with ~15% overlap. Each chunk keeps `start`/`end` timestamps.
- PDFs/docs: split by heading, then paragraph, targeting ~400 tokens.
- Every chunk gets `{course, source_path, source_type, week, topic, page|slide|t_start,t_end, chunk_id}`.
- **[new] Contextual header.** Prepend a one-line header to the embedded text (`"COMP3000 · Week 3 · Lecture 2 – Graph search · slide 14"`). It noticeably improves retrieval for short or ambiguous chunks and costs nothing.

## 3. Retrieval [changed]

- **LanceDB over Chroma.** LanceDB has built-in full-text search, which makes **hybrid search** (vector + BM25-style keyword) easy. Lecture material is full of exact terms, acronyms, formula names, and code identifiers that pure embeddings miss.
- **Embeddings: `BAAI/bge-small-en-v1.5` instead of `all-MiniLM-L6-v2`, run locally via `fastembed` (ONNX runtime).** [changed] It's similarly small and fast on CPU and clearly better at retrieval — MiniLM truncates at 256 word-pieces, which silently cuts longer chunks. Running it through `fastembed` instead of PyTorch means no CUDA install is required on a fresh Windows machine. The embedder is behind an interface, so Voyage or a larger local model is a config change.
- **Hybrid (vector + keyword, combined with RRF) is the default retrieval mode.** [decided] The first real eval run against the COMP4650 course (28 answerable test questions) backs this: hybrid hit@1/3/10 = 0.57/0.89/0.93, MRR 0.71 — matching or beating vector-only (0.61/0.82/0.89, MRR 0.71) and keyword-only (0.57/0.79/0.93, MRR 0.70) on the metric that matters most for chat (hit@3), while staying close to the best of either alone everywhere else. See `docs/LEARN.md` Phase 2 for the full write-up including what the early "misses" turned out to be (a labelling bug in the eval harness, not retrieval).
- **Optional reranker** (`bge-reranker-base` via `fastembed`, local) over the top ~30 hybrid hits → top 8–12 sent to Claude. **Off by default** — wired up and ready, but the hybrid numbers above are already strong enough that it isn't needed yet. Revisit if a future eval set shows it helps.
- **Filters:** "only week 3", "only the tutorials", "only this file".

## 4. Grounded chat

- Retrieved chunks are sent to Claude as **document blocks with native citations enabled**. Claude's response comes back with structured citations pointing at exact cited text, so we map each one to *file + page/slide/timestamp* reliably instead of parsing `[1]` markers out of prose.
- **Sources-only mode:** system prompt requires answering only from the provided documents, and when they don't contain the answer Claude must say so and give the closest related material it did find.
- **Open mode:** same retrieval, plus Claude's own knowledge and the server-side **web search tool**. The UI labels which parts came from your notes, which came from the web, and which are uncited model knowledge.
- **[new] "Deep" mode (full-context).** Claude now has a **1M-token context window** on current models, so a week's worth of material (or a whole course for many units) can go in **directly**, cached with **prompt caching** so follow-up questions are cheap. This works better than top-k for "compare lecture 2 and lecture 5" or "what are all the assumptions in this unit" questions, where retrieval tends to miss pieces. Default stays RAG, and Deep mode is a toggle.
- Multi-turn: keep chat history. Rewrite follow-ups ("what about the second one?") into standalone queries before retrieval.
- Model IDs are in config. Defaults (cheapest profile): **Sonnet 5** (`claude-sonnet-5`, $2/$10 per M tokens) for chat, audio scripts and Deep mode; **Haiku 4.5** (`claude-haiku-4-5`) for query rewriting and helper calls. Opus is optional.

## 5. Audio overview [changed: coverage-first, adaptive length]

Top-k retrieval is the wrong tool for summaries, because an overview needs **coverage**, not similarity. The pipeline is built so that nothing important gets skipped:

1. **Scope:** a week, topic, file set, or whole course, plus an optional focus ("exam prep", "just the maths").
2. **Key-point extraction (map, Haiku 4.5, one call per source):** list every concept, definition, method, worked example and caveat, each with an **importance score** and evidence:
   - explicit emphasis ("this will be on the exam", "the key idea is…", boxed/bold slide titles)
   - time the lecturer spends on it in the transcript
   - whether it appears in **both** the slides and the transcript (strong signal)
   - assessment mentions (quiz/assignment topics)
3. **Merge + rank (Sonnet 5):** dedupe across slides and transcripts into one ranked list. Tier A = must cover in depth, Tier B = cover briefly, Tier C = mention or skip.
4. **Adaptive length:** budget ≈ 90 s per Tier A point + 30 s per Tier B + 1 min intro/outro, **clamped to 5–45 min** (≈150 spoken words/min). If the full budget exceeds 45 min, B points are compressed first, and A points are never dropped. The user can override the length.
5. **Outline into chapters,** then **script chapter by chapter** (Sonnet 5), each call given the full scope material plus the running summary so the hosts stay consistent. Output is structured JSON `[{speaker, text, sources}]`. Two hosts: one explains, one asks the questions a student would ask, including likely confusions.
6. **Coverage check (Haiku 4.5):** verify every Tier A point is actually explained in the script, then regenerate the chapter for any that are missed. Produce a coverage report.
7. **TTS:** Kokoro, via **`kokoro-onnx`** (ONNX runtime) rather than the PyTorch Kokoro package. [planned, to validate in Phase 6] Same reasoning as the embedder: no CUDA/PyTorch install needed on a fresh Windows machine. Two distinct voices, per-line synthesis, stitched with `pydub` (+ `ffmpeg`) with natural pauses → MP3 plus a transcript with chapter markers and source links.

## [new] AgentAUS provider (branch)

**Decision:** add AgentAUS, a sovereign Australian model from Trellis Data
with an OpenAI-compatible API, as an alternative LLM provider, and make it
the default on this branch. Claude stays available as a config switch
(`NOTECAST_PROVIDER=anthropic`). Retrieval (fastembed) and audio TTS
(Kokoro) are unchanged either way — they never call an LLM. The owner has
now measured the real endpoint, so the defaults in `notecast/config.py` are
the real values, not placeholders: base URL
`https://agentaus.com.au/api/v1` (the `/api` matters — plain `/v1` redirects
to a login page), model `agentaus.v1` (the only one `/models` lists; the
server accepts any model name and just replies as "agentaus"), context
window 131,072 tokens (input + output).

The adapter (`notecast/providers/openai_compat.py`, `OpenAICompatClient`)
presents the same interface the rest of NoteCast expects from the Claude
client, so chat, retrieval, and audio scripting code don't need to branch
on provider. It emulates the features AgentAUS's API doesn't have natively:
citations via numbered `<source id="n">` prompts and `[n]` markers mapped
back to file + page/slide/timestamp, and structured JSON output (audio key
points, scripts, coverage check) via a described JSON Schema plus
`response_format={"type": "json_object"}` when
`NOTECAST_AGENTAUS_JSON_MODE=true`. AgentAUS accepts that parameter but
doesn't enforce it, so the real safety net is that NoteCast validates every
JSON reply and retries once with a "JSON only" instruction; the
automatic-fallback-if-rejected behaviour is still there for other
OpenAI-compatible servers, it just isn't what's doing the work here. Every
call also sends `system_prompt_overwrite: true`
(`NOTECAST_AGENTAUS_SYSTEM_PROMPT_OVERWRITE`, default on), replacing
Trellis's hidden ~2,400-token default system prompt with NoteCast's own
(~130 tokens), which matters a lot for cost and latency since there's no
prompt caching. No `<think>` tags have actually been observed from AgentAUS
(its reasoning is hidden but still counts toward output tokens); NoteCast
strips the rare leaks it has seen instead — leading plain-text reasoning
lines and raw control tokens. A new `notecast provider-check` command
prints the active provider/models, lists what the endpoint offers, and
sends a small ping request, for confirming the key/base URL/model ID.

**Known trade-offs vs Claude** (see [`docs/AGENTAUS.md`](AGENTAUS.md) for
the full table):
- Citations are emulated, not native — the cited span is the whole source
  excerpt rather than the exact cited sentence.
- No web search tool of NoteCast's own in open mode; open mode with
  AgentAUS is sources + model knowledge only. (AgentAUS does have its own
  built-in web search, which can switch on uninvited even with
  `tool_choice: none` and add 20k–100k input tokens to a call — a known
  cause of occasional slow calls or context-overflow errors, not something
  NoteCast controls.)
- No prompt caching, so deep mode resends the full scope every turn —
  slower and pricier per follow-up than with Claude.
- Deep mode is capped at `NOTECAST_AGENTAUS_CONTEXT_TOKENS` (131,072 tokens,
  input + output; deep mode budgets to roughly 119k tokens of material)
  instead of Claude's ~1M-token window.
- Output length isn't actually controlled by
  `NOTECAST_AGENTAUS_MAX_OUTPUT_TOKENS` — AgentAUS ignores `max_tokens` and
  self-limits every reply to roughly 2,500 words (~100 tokens/sec; longest
  observed ~6.7k tokens including hidden reasoning). NoteCast's audio
  chapters (~850 words, up to 8 for a 45-minute episode) and a compact
  merge-and-rank output format for AgentAUS are sized to fit under that;
  pushing `NOTECAST_AUDIO_MAX_MINUTES` well past 45 risks chapters that
  don't fit in one reply.
- No cheaper helper model exists — `/models` lists exactly one model, so
  `NOTECAST_AGENTAUS_HELPER_MODEL` has nothing to point at yet.
- Cost estimates show `n/a` unless you set
  `NOTECAST_AGENTAUS_PRICE_INPUT_PER_MTOK` / `..._OUTPUT_PER_MTOK` yourself
  — AgentAUS pricing isn't published, and the owner has unlimited usage on
  the current key, so this is expected rather than a gap to fix.

## 6. [new] Evaluation (small but essential)

A `tests/eval/` set of ~20–30 question/answer/source triples per test course, including **questions whose answer is *not* in the material**. We track:
- retrieval hit rate (was the right chunk in top-k?)
- refusal correctness in sources-only mode
- citation validity (does the cited text actually support the claim?)

This is how we'll know if hybrid search, the reranker, or chunk size changes actually help.

## 7. Interface: Streamlit

- Sidebar: notebook picker, create a notebook, drag-and-drop upload, ingest button with progress, mode toggle (Sources only / Open / Deep), scope filters (week, file type).
- Main pane: chat with inline citation chips. Clicking one shows the source excerpt with its page, slide or timestamp.
- Audio tab: pick scope, format and length, generate, then play or download with the transcript.
- A thin CLI (`notecast ingest|chat|audio`) sits on the same core for scripting.

---

## Build phases (each ends with verify → commit → push)

Parallel tracks are in brackets; they run as simultaneous Sonnet builders.

| Phase | Deliverable | Parallel tracks | Status |
|---|---|---|---|
| **0. Scaffold** | `pyproject` (uv), ruff, pytest, config, `.env.example`, core data models (`Chunk`, `Source`, `Notebook`), and interfaces for parser/embedder/store | [scaffold+core] [Windows setup + LEARN docs] | Done |
| **1. Ingestion** | all parsers + chunker + manifest/incremental ingest | [PDF] [PPTX+DOCX] [VTT/SRT/TXT] [chunker+manifest] | Done |
| **2. Index + retrieval** | embedder, LanceDB store, hybrid search, filters | [embedder] [store+hybrid] [eval harness] | Done |
| **3. Grounded chat** | Claude client with citations, sources-only mode, citation → location mapping, CLI chat | [claude client+prompts] [citation mapper] [CLI] | Done |
| **4. Modes** | open mode + web search, Deep (full-context + caching) mode | [web search] [deep mode] | Done |
| **5. Multi-notebook** | create/list/switch/delete courses, scope filters | single builder | Done |
| **6. Audio overview** | key points → rank → adaptive outline → chapter scripts → coverage check → Kokoro TTS → stitch | [key points+ranking] [script+coverage] [TTS+stitch] | Done |
| **7. UI** | chosen front end (Streamlit) | depends on choice | Done — verified with a fake Claude; real-API run pending |
| **8. Extras** | quiz/flashcards/study guide; later maybe Whisper transcription and OCR | parallel per feature | Not started |

Phase 1 must be solid before anything else, same as the original plan.

## What's next / open questions

- **Real end-to-end test with an API key.** Everything through Phase 6 has
  been unit-tested with the Claude API, embedder, and TTS mocked. Nothing
  has yet been run against the live Anthropic API by a human with a real
  key and real course material end to end (ingest → ask/chat in all three
  modes → generate an audio overview and actually listen to it).
- **Tune audio prompts after listening.** The coverage-first pipeline is
  built and unit-tested, but the *quality* of the generated hosts' banter,
  pacing and tone can only really be judged by listening to a real
  generated episode and adjusting `notecast/audio/prompts.py` from there.
- **Optional reranker test on the owner's PC.** `rerank_enabled` is wired up
  and off by default (see Phase 2 notes in `docs/LEARN.md`); worth an eval
  run on a second real course to see if it earns its keep before ever
  turning it on by default.
- **Quiz / flashcards (Phase 8).** Not started; see the Phase 8 idea list in
  `docs/LEARN.md`.
- **Live test against AgentAUS.** The owner has measured the real endpoint
  (context window, output behaviour, JSON mode, system prompt overwrite —
  see the section above and `notecast/config.py`), so context window, max
  output tokens, JSON mode support, and pricing (unpublished, unlimited
  usage on the current key, so left unset by design) are answered. What's
  still pending is a full live end-to-end run with the owner's real key —
  ingest → ask/chat in all three modes → generate and listen to an audio
  overview — against AgentAUS specifically, the way Phase 7 still needs for
  Claude.
