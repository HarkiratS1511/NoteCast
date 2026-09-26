# Using NoteCast

A day-to-day guide to actually using NoteCast once it's installed. If you
haven't installed it yet, start with
[`docs/SETUP-WINDOWS.md`](SETUP-WINDOWS.md). If you want to understand *why*
it's built this way, see [`docs/LEARN.md`](LEARN.md). This page is about
*what to type*.

All commands below are run from the project folder, as `uv run notecast ...`.
Run `uv run notecast --help` or `uv run notecast <command> --help` any time
— this guide mirrors the real `--help` output, but the CLI is always the
source of truth.

## 1. Add a course

A "notebook" is one course's folder.

```bash
uv run notecast notebooks create comp4650-document-analysis
uv run notecast notebooks list
```

Then copy your files into `notebooks/<slug>/sources/`, ideally organised by
week so NoteCast can infer week numbers and you can filter by week later:

```
notebooks/comp4650-document-analysis/sources/
  week-01/
    lecture-1-slides.pdf
    lecture-1-transcript.vtt
  week-02/
    ...
```

Supported file types: PDF, PPTX, DOCX, VTT, SRT, TXT, MD. Nesting is
flexible — `week-01/`, `wk1`, `topic-graphs/` are all recognised, and you can
override the week/topic for a file in `notebooks/<slug>/course.yaml` if the
folder name doesn't match.

This folder (`notebooks/`) is git-ignored on purpose — see the
[Privacy note in the README](../README.md#privacy). Your course material
never gets committed and only the specific passages a question needs are
ever sent to Claude.

## 2. Ingest (parse + index)

```bash
uv run notecast ingest comp4650-document-analysis
```

This parses every file in `sources/`, splits it into chunks (one slide, one
paragraph, one ~60–90s transcript window — see LEARN.md for why), embeds
them, and builds the search index.

Options:
- `--force` — reprocess every file even if unchanged (normally NoteCast
  skips files whose content hasn't changed, via a hash).
- `--reindex` — rebuild the search index from scratch after ingesting.

Run this again whenever you add or change files in `sources/` — only the
new/changed files get reprocessed.

**First run only:** downloads the embedding model (~70 MB) into
`.cache/models/`. Needs internet once; after that it's read from disk.

## 3. Search (no Claude, no cost)

To sanity-check the index directly, without calling Claude:

```bash
uv run notecast search comp4650-document-analysis "TF-IDF"
```

Options: `-k <int>` (results to show, default 5), `--week <int>` (repeatable,
restrict to a week), `--mode hybrid|vector|keyword` (default `hybrid`).

This is free (no API calls) and useful for checking retrieval quality or
just finding "which slide talked about X" quickly.

## 4. Ask a question / chat

**One-off question:**

```bash
uv run notecast ask comp4650-document-analysis "What is Laplace smoothing?"
```

**Interactive chat** (keeps history, rewrites follow-ups like "what about the
second one?" into standalone questions before searching):

```bash
uv run notecast chat comp4650-document-analysis
```

Both take:
- `--mode sources|open|deep` (default `sources`)
- `--week <int>` (repeatable)
- `-k <int>` — number of chunks to retrieve (`ask` only)
- `--yes` / `-y` — skip the confirmation prompt before spending on deep mode

### Modes, and when to use which

- **`sources`** (default): Claude answers *only* from the top-k retrieved
  chunks for your question. Cheap, fast, tightly cited. If the retrieved
  material doesn't actually answer the question, the reply starts with the
  marker `[NOT_IN_SOURCES]` — shown to you as **"Not in your course
  material"** — followed by an honest "the material doesn't cover this,"
  plus the closest related thing it did find, if any. This is the safe
  default: it never quietly makes something up to fill a gap.
- **`open`**: same retrieval, but Claude may also use its own general
  knowledge and a web search tool. The answer clearly separates what came
  from your course material, what came from the web, and what's the model's
  own uncited knowledge — it's not supposed to blur these together.
- **`deep`**: skips top-k search entirely and gives Claude the *whole*
  scope (a week, or the whole course) as context, so it can compare things
  across lectures or answer "what are all the assumptions in this unit"
  style questions that top-k retrieval tends to miss pieces of. This costs
  real money and shows you an estimate first — see the cost cheat-sheet
  below — and asks "Continue?" unless you pass `--yes`.

**Citations:** every answer is grounded with Claude's native citation
feature — it points at the exact passage it used, and NoteCast maps that
back to the file, plus the slide/page/timestamp, so you can always check
the source yourself instead of taking the answer on faith.

## 5. Audio overviews

Generate a two-host, podcast-style overview of a week, a file, or the whole
course:

```bash
uv run notecast audio comp4650-document-analysis --week 3
```

Options:
- `--week <int>` (repeatable) — restrict to specific weeks
- `--source <str>` (repeatable) — restrict to specific source file paths
- `--focus <str>` — e.g. `--focus "exam prep"` or `--focus "just the maths"`
- `--script-only` — write the script but don't render audio (cheapest way to
  check what it *would* say before spending on TTS rendering — though
  rendering itself is free, see below)
- `--minutes-max <int>` — override the maximum length for this run
- `--yes` / `-y` — skip the cost confirmation prompt

**What happens, in five steps** (see LEARN.md Phase 6 for the full story):
1. Every source in scope has its key points extracted (concepts, definitions,
   worked examples, caveats), each scored for importance.
2. The points are merged and ranked into tiers: must-cover, cover-briefly,
   mention-or-skip.
3. A time budget is worked out from how many top-tier points there are
   (clamped between `NOTECAST_AUDIO_MIN_MINUTES` and
   `NOTECAST_AUDIO_MAX_MINUTES`, default 5–45 min).
4. The episode is outlined into chapters and scripted chapter by chapter,
   two hosts (one explains, one asks the questions a student would ask).
5. A coverage check verifies every must-cover point actually made it into
   the script, and regenerates any chapter that missed one, before the
   script is turned into audio with Kokoro (local, free, CPU).

Before any of this runs, NoteCast prints an **estimated cost** (with a
range) and asks "Continue?" unless you pass `--yes`.

**Re-rendering for free:** every run also saves the generated script as
JSON. If you just want to change the voice, speed, or bitrate without
regenerating the script (no Claude calls, no cost), use:

```bash
uv run notecast audio-render comp4650-document-analysis notebooks/comp4650-document-analysis/audio/week-03-20260926-1400.script.json
```

**Where files go:** `notebooks/<slug>/audio/` — the script JSON, a
transcript markdown with chapter markers and source links, and the
rendered MP3.

**First audio run only:** downloads the Kokoro voice model (~350 MB, model
+ voices) into `.cache/models/kokoro`, once. No `ffmpeg` install needed —
NoteCast stitches the MP3 itself.

**Voices:** two hosts, distinct voices by default — Host A `af_heart` (US),
Host B `bm_george` (UK). Change either with the environment variables
`NOTECAST_HOST_A_VOICE` / `NOTECAST_HOST_B_VOICE` in your `.env`. Other
English voices available: `af_alloy`, `af_aoede`, `af_bella`, `af_heart`,
`af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`,
`af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`,
`am_michael`, `am_onyx`, `am_puck`, `am_santa`, `bf_alice`, `bf_emma`,
`bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`.

Rendering itself runs on CPU and is roughly 3x faster than real time — a
45-minute episode takes about 14 minutes to render.

## 6. Web UI

```bash
uv run notecast ui
```

Opens a local browser tab: notebook picker, ingest button, chat with mode
toggle and inline citations, and an audio tab to generate and play/download
overviews. It's the same core as the CLI, just a friendlier front end — any
step that spends API money still shows a cost estimate and needs a
confirmation click, the same way the CLI does. (This part of NoteCast is
newer than the rest and still being checked over, so if something looks
off, the CLI commands above are the well-tested fallback.)

## Costs cheat-sheet

Real numbers from testing against a COMP4650 course (whole course ≈ 47k
tokens; week 3 alone ≈ 31k tokens):

| Action | Typical cost |
|---|---|
| `search` | Free — no API call |
| `ask` / `chat`, mode `sources` or `open` | A few cents at most — only top-k chunks are sent |
| `chat`, mode `deep`, whole course, first question | ≈ $0.12 |
| `chat`, mode `deep`, whole course, each follow-up (cache hit) | ≈ $0.01 |
| `audio`, week 3 | ≈ $0.33 (range $0.25–$0.40) |
| `audio`, whole course | ≈ $0.42 |
| `audio-render` (re-render a saved script) | Free — no Claude calls |
| Rendering audio (TTS) | Free — runs locally on your CPU |

Deep mode is cheap here because prompt caching means you only pay full price
once per scope; every follow-up question in the same scope reads the cached
material instead of resending it. This is specific to how small a single
course's material is (tens of thousands of tokens) — see LEARN.md Phase 4.

## Troubleshooting

**"Missing API key" / chat or audio commands fail immediately**
You need an API key for whichever LLM provider is configured. By default
that's AgentAUS (`NOTECAST_PROVIDER=agentaus`), needing `AGENTAUS_API_KEY`
and `AGENTAUS_BASE_URL` in `.env` — see
[`docs/AGENTAUS.md`](AGENTAUS.md). With `NOTECAST_PROVIDER=anthropic` you
need `ANTHROPIC_API_KEY=...` instead; see
[`docs/SETUP-WINDOWS.md`](SETUP-WINDOWS.md#getting-an-anthropic-api-key) for
where to get one. `search` and `ingest` don't need a key — only commands
that call the LLM do. Run `uv run notecast provider-check` to confirm which
provider is active and that its key/base URL/model ID actually work.

**"Index not built" / search returns nothing / IndexNotBuiltError**
Run `uv run notecast ingest <slug>` first. If you've already ingested and
still get this, try `uv run notecast ingest <slug> --reindex`.

**Model download fails (embedding or Kokoro voice model)**
Both are one-time downloads (~70 MB for embeddings, ~350 MB for Kokoro) that
need internet the first time only. If it fails partway, check your
connection and just re-run the same command — it'll retry the download.
Corporate/university networks sometimes block large downloads; try a
different network if it keeps failing.

**"Nothing in scope" (deep mode or audio)**
Your `--week` filter (or the notebook itself) doesn't match any ingested
material. Run `uv run notecast search <slug> "" -k 1` or
`uv run notecast notebooks list` to check the notebook has files, and that
you ingested it, and that the week numbers you're filtering on actually
exist (check the folder names under `sources/`).

**Answer says "Not in your course material" / `[NOT_IN_SOURCES]`**
This is `sources` mode being honest: the top-k chunks it retrieved for your
question didn't contain an answer. Try rephrasing, widening the `--week`
filter, using `--mode open` to let Claude use general knowledge too, or
`--mode deep` if the answer requires connecting several lectures.

**Streamlit UI won't start / import errors**
Make sure `uv sync` has been run recently (`uv sync` picks up new
dependencies). The UI is still being verified — if the CLI commands above
work but the UI doesn't, that's a known rough edge, not something wrong
with your setup.
