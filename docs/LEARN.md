# How NoteCast works — learning notes

This is a living document. It grows one section per build phase, explaining
the concepts behind what just got built, in plain English. If you're new to
some of this (APIs, virtual environments, embeddings, RAG), start here.

---

## Phase 0: Scaffold

### The big picture

NoteCast's two main jobs — answering questions and generating audio
overviews — both start from the same pipeline: turn your files into small,
searchable pieces, then hand Claude only the relevant pieces.

```
 your files (PDFs, slides, transcripts)
        │
        ▼
   parsed text            (extract raw text + page/slide/timestamp)
        │
        ▼
     chunks               (split into small labelled pieces, e.g. "one slide")
        │
        ▼
  search index            (so we can find the *relevant* chunks fast)
        │
        ▼
question ──► relevant chunks + your question ──► Claude
        │
        ▼
   answer, with citations back to the exact file/page/slide/timestamp


 -- for audio overviews --

 key points (extracted from chunks) ──► script (two hosts) ──► TTS voices ──► stitched MP3
```

### Why not just paste everything into Claude's chat?

A few reasons:

- **Attachment limits.** Claude's chat app limits how much you can upload at
  once. The **API** (see glossary below) has no such UI limit — Claude models
  can actually read very large amounts of text — but there's still a reason
  to be selective.
- **Retrieval sends only what's relevant.** If you ask "what's the formula for
  TF-IDF?", NoteCast finds the handful of chunks that actually mention it and
  sends just those, instead of your entire course. That's faster, cheaper,
  and it also makes Claude's citations far more precise, because it's
  choosing from a small, relevant set rather than skimming everything.
- **Cost.** Claude's API is billed per **token** (see glossary). Sending your
  whole course on every single question adds up fast; sending only the
  relevant few chunks doesn't. (NoteCast's "Deep" mode, added in a later
  phase, deliberately sends everything for questions where that's actually
  the better tool — more on that when we build it.)

### Glossary

- **API (Application Programming Interface) / API key** — a way for one
  program to ask another program to do something, over the internet, instead
  of a human clicking a UI. Here, NoteCast's code talks to Claude's API to get
  answers and generate scripts. An **API key** is like a password that proves
  the request is coming from you, so Anthropic can bill and rate-limit
  correctly. Keep it secret — anyone with your key can spend your credits.
- **Virtual environment** — an isolated folder (NoteCast's is `.venv`) holding
  just the Python packages this one project needs, at the versions it needs.
  Without it, different projects on your machine could end up fighting over
  conflicting versions of the same package.
- **Package manager (`uv`)** — the tool that reads the project's list of
  dependencies, downloads them, and installs them into the virtual
  environment. It also installs the right version of Python itself.
- **Dependency** — a piece of code someone else wrote that NoteCast relies on
  instead of reimplementing itself (e.g. a PDF-reading library). Listed in
  `pyproject.toml`.
- **CLI (Command Line Interface)** — a program you control by typing text
  commands instead of clicking buttons. `uv run notecast ...` runs NoteCast's
  CLI.
- **git commit / push** — a **commit** is a saved snapshot of your code
  changes with a message describing them. A **push** uploads your commits to
  GitHub so others (or you, on another machine) can get them.
- **`.gitignore`** — a file listing things Git should never track or upload
  (e.g. your `.env`, your course material, generated audio). Keeps secrets
  and copyrighted material out of the public repo.
- **`.env`** — a plain text file of settings and secrets read by the app at
  startup, kept out of Git by `.gitignore`. `.env.example` is the public
  template showing which settings exist, without real values.
- **Tokens** — the units Claude (and most language models) read and bill by.
  Roughly: **1 token ≈ ¾ of an English word** (so ~100 tokens ≈ 75 words).
  Longer input and output costs more tokens, hence more money and time.
- **Embeddings** — think of them as "meaning coordinates": a piece of text
  gets converted into a list of numbers (a vector) such that texts with
  similar *meaning* end up at nearby coordinates, even if they don't share
  exact words. This is what lets NoteCast find "the formula for measuring how
  important a word is in a document" when you search for "TF-IDF", without
  the words needing to match exactly.
- **Vector database** — a database built to store embeddings and quickly find
  the nearest ones to a given query — i.e. "which chunks mean something
  similar to this question?". NoteCast uses LanceDB.
- **Chunk** — one small, self-contained piece of your source material (e.g.
  one slide, or a 60-second window of a transcript), small enough to embed
  and retrieve individually, with metadata (file, page/slide/timestamp)
  attached.
- **RAG (Retrieval-Augmented Generation)** — the overall technique of
  *retrieving* the relevant chunks first, then *generating* an answer with
  those chunks as context, instead of relying only on what the model already
  "knows". It's why NoteCast's answers can be grounded in your specific
  course material rather than generic knowledge.
- **Citation** — a pointer from a claim in Claude's answer back to the exact
  chunk (and file/page/slide/timestamp) that supports it, so you can verify
  it yourself.

### What's in the repo (a short tour)

```
notecast/
  config.py       settings — reads .env, holds model IDs, paths, defaults
  models.py       the data shapes — e.g. what a "Chunk" or "Source" looks like
  interfaces.py   contracts — abstract definitions that later pieces (parsers,
                  embedders, stores) plug into, without needing to know
                  each other's internals
  notebook.py     folder management — creating/listing notebooks on disk
  cli.py          terminal commands — what `uv run notecast ...` runs
tests/            automated tests, one file per module roughly
docs/
  PLAN.md         the architecture and build-phase plan
  LEARN.md        this file
  SETUP-WINDOWS.md   the Windows install guide
```

### Phase log

**Phase 0 — Scaffold.** Built the project skeleton: dependency setup (`uv`),
linting/formatting (`ruff`), the test runner (`pytest`), configuration
loading, the core data shapes (`Chunk`, `Source`, `Notebook`), and — the
interesting part — **interfaces** for the parser, embedder, and store that
later phases will implement.

*One concept worth remembering:* an **interface** is a promise about *what a
piece of code will do* (its inputs and outputs) without saying *how*. Because
the parser, embedder, and storage layers all plug into agreed interfaces,
different people (or, here, different AI builders) could build the PDF
parser, the PPTX parser, and the embedder at the same time, in parallel,
without needing to see each other's code — they just need to honour the
same contract. This is the same reason large software teams split up work:
agree on the shape of the connection first, then build both sides
independently.

**Phase 1 — Ingestion.** Coming soon.

**Phase 2 — Index + retrieval.** Coming soon.

**Phase 3 — Grounded chat.** Coming soon.

**Phase 4 — Modes (Open / Deep).** Coming soon.

**Phase 5 — Multi-notebook.** Coming soon.

**Phase 6 — Audio overview.** Coming soon.

**Phase 7 — UI.** Coming soon.

**Phase 8 — Extras.** Coming soon.
