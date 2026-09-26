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

**Phase 1 — Ingestion.** Built the parsers, the chunker, and incremental
ingest (`uv run notecast ingest <slug>`).

A **parser** is the piece of code that turns one file format into plain
text plus location metadata (which page, which slide, which timestamp).
NoteCast has one parser per format — PDF, PPTX, DOCX, VTT/SRT, TXT/MD — and
they all produce the same shape (a `ParsedDocument` full of `Section`s), so
everything downstream (chunking, embedding) doesn't need to know or care
which parser produced a given piece of text.

*Slide PDFs, one slide per chunk.* Course slides are usually PDFs exported
from PowerPoint. The PDF parser detects this by page shape: if most pages
are **landscape** (wider than tall), it treats the file as a slide deck and
labels each page "slide N" instead of "page N". Each slide becomes its own
chunk (instead of, say, splitting by paragraph) because that's the natural
unit a student thinks in — "what does slide 14 say" — and it keeps citations
precise: NoteCast can point you at one specific slide rather than a vague
range.

*The footer-stripping bug.* Every slide in the real course PDFs has a
repeated footer: "ANU SCHOOL OF COMPUTING | DOCUMENT ANALYSIS" plus a page
number. Left in, that footer would pollute every single chunk and get
picked up by keyword search as if it were meaningful content. The fix:
any line that repeats on more than half the pages counts as boilerplate and
gets removed. The first version of this rule also collapsed digits (so
"page 9" and "page 12" would count as the same repeating line, catching
page numbers with different values). That was too aggressive — it also
collapsed things like "Example 1", "Example 2", "Example 3" on different
slides into the same "Example #" key and wiped them out too, even though
they were real slide content, not boilerplate. The fix was to only allow
that digit-insensitive matching for lines that are *small* (font size at or
below the page's median) and sitting in the top/bottom **12%** of the page
— the header/footer zone. A big, numbered slide title sitting in that same
zone has to match some other boilerplate line *exactly* (not digit-collapsed)
to be treated as repeated, so numbered body content survives.

*Cleaning up the transcript.* Lecture transcripts are full of spoken-word
noise that doesn't help search or reading: filler words ("uh", "um", "er")
and stutters ("I, I, I told you"). The cleanup step removes standalone
fillers and collapses immediate repeats. The tricky part: after removing a
filler, two words that used to be separated by it can end up sitting right
next to each other, and a naive rule would then treat them as a stutter too.
That caused a real bug: "after all, uh, all these" — where "all" appears
twice for two different reasons, not because the lecturer stuttered — was
getting collapsed into "after all these", silently changing the meaning.
The fix narrows that second pass to a short, curated list of function words
that actually do get stuttered this way in speech (pronouns, articles,
conjunctions, auxiliaries, prepositions — "we", "the", "will", "to", and so
on). "All" isn't on that list, so it's never collapsed across a removed
filler, only when it's *already* adjacent in the raw text (a genuine "all,
all these" stutter with nothing removed in between).

*Why transcripts get "≈ 23 min" labels.* Plain-text transcripts (as opposed
to VTT/SRT caption files) don't have timestamps — just blocks of speech.
NoteCast estimates a position by counting words from the start and dividing
by a typical lecture speaking rate (150 words/minute), so a citation can
still say roughly where in the recording something was said, even without
exact timing.

*Chunking.* Slides and transcript windows target **~350 tokens** per chunk
(capped at 500), because that's small enough for a search result to be
genuinely focused on one topic, but large enough to hold a complete idea
without slicing a sentence in half. Transcript windows overlap by ~15% so
an idea that spans a window boundary isn't lost to either side. Title-only
or near-empty slides ("Questions?", section dividers) get folded into a
neighbouring slide instead of becoming their own tiny, low-value chunk — and
when that happens, the merged chunk's label becomes a range like "slides
4–6" rather than just the first slide, so the citation still tells you
where to look.

*Contextual headers.* Before a chunk gets embedded, NoteCast prepends a
one-line header like "COMP4650 · Week 3 · Lecture 2 · slide 14" to the text
that actually gets turned into a vector (not to the text you'd read). Short
or ambiguous chunks — a single formula, a one-word slide title — are much
easier to find with that context attached, and it's essentially free.

*Incremental ingest.* Re-running `notecast ingest` on a notebook doesn't
reprocess every file from scratch. Each source file's contents are hashed
(SHA-256, a fingerprint that changes if even one byte changes), and the
hash is recorded in `manifest.json`. On the next run, unchanged files are
skipped entirely, changed files are reprocessed, and files you deleted are
removed from the index. This matters once you're adding a lecture every
week instead of dropping in a whole course at once.

*Why double verification.* Every phase in this project is built by one AI
"builder" and then checked by a separate AI "verifier" that didn't write
the code and re-runs the tests itself, adversarially. This isn't
redundancy for its own sake — the footer-stripping bug above (and the
transcript stutter bug) are exactly the kind of thing a builder's own tests
can miss, because the builder wrote both the code and the tests with the
same blind spot in mind ("of course collapsing digits is fine — I only
tested it on footers"). A second reader, looking only at behavior and
edge cases, catches what the original author didn't think to check.

**Phase 2 — Index + retrieval.** Built the embedder, the LanceDB store,
hybrid search, and the retrieval eval harness.

*Embeddings, with a concrete example.* An embedding turns a chunk of text
into a list of numbers (a vector) such that texts with similar *meaning*
end up at nearby coordinates — even when they don't share exact words. In
one of our own retrieval tests, the chunk about **"Laplace smoothing"**
scored **0.81 similarity** against a query about **"add-one smoothing"**
(the same technique, different name) but only **0.61** against a query
about **"PageRank"** (an unrelated topic). That gap is the whole point:
embeddings let you search by what something *means*, not just what words it
uses.

*Keyword search (BM25), and why exact terms still matter.* Embeddings are
great at meaning but can be surprisingly bad at exact terms: acronyms,
formula names, variable names, anything where the *precise string* matters
more than the concept around it. Searching for "TF-IDF" or "BM25" itself
should reliably find the slide that says "TF-IDF", not just something
vaguely about relevance. Keyword search (the classic **BM25** ranking
algorithm) handles that: it scores chunks by how well their exact words
match the query's exact words.

*Hybrid search = both, combined.* Neither approach alone is enough, so
NoteCast runs both a vector search and a keyword search for every query and
merges the two ranked lists with **Reciprocal Rank Fusion (RRF)** — a
simple, well-tested way to combine two rankings: a chunk that ranks well in
*either* list (and especially one that ranks well in *both*) rises to the
top of the merged list. Hybrid search is now the default mode, and the eval
numbers below are why.

*Why fastembed, why LanceDB.* NoteCast uses **fastembed** to run the
embedding model (`BAAI/bge-small-en-v1.5`) locally. fastembed runs models in
the **ONNX** runtime instead of PyTorch, which means no CUDA/GPU install is
needed to get good speed on Windows — one less multi-gigabyte, easy-to-get-
wrong install for a fresh machine. **LanceDB** is the vector database:
it's *embedded* (a library your program loads directly, not a server you
have to run and manage separately) and has full-text/keyword search built
in, which is exactly the second half of hybrid search.

*Rerankers (optional, off by default).* A reranker is a second, more
expensive model that re-scores a shortlist of hits (the top ~30) for one
specific query, rather than pre-computing a generic embedding for each
chunk. It can improve ranking quality but costs extra time per query.
NoteCast has one wired up (`bge-reranker-base` via fastembed) but leaves it
off by default until an eval run shows it's worth the cost.

*Evaluation: hit@k and MRR, explained simply.* To know whether any of this
actually works, NoteCast runs a set of real questions against the real
course material and checks: **hit@k** — for what fraction of questions was
the correct chunk somewhere in the top *k* results? (hit@1 = "was it the
very first result", hit@10 = "was it anywhere in the top 10"). **MRR**
(Mean Reciprocal Rank) — on average, *how far down* the list was the right
answer, expressed as 1/rank averaged across all questions (1.0 = always
first, 0.5 = on average around 2nd, and so on).

*The first real results,* on 28 answerable test questions written against
the actual COMP4650 slides and transcript:

| Mode | hit@1 | hit@3 | hit@10 | MRR |
|---|---|---|---|---|
| **Hybrid (default)** | 0.57 | 0.89 | 0.93 | 0.71 |
| Vector only | 0.61 | 0.82 | 0.89 | 0.71 |
| Keyword only | 0.57 | 0.79 | 0.93 | 0.70 |

Hybrid wins where it matters most for a chat product: by hit@3, it already
finds the right chunk 89% of the time, ahead of either method alone, and it
stays close to the best of both on every other measure. This is the
evidence behind making hybrid the default.

*What the misses taught us.* The first eval run showed several "misses"
that turned out not to be retrieval failures at all: the correct slide
*was* being returned, but the eval case expected it labelled as, say,
"slide 5", while the tiny-slide-merging described above had folded slides
4–6 into one chunk labelled with a range. The eval's matching logic only
checked a single slide number, so a technically-correct hit was being
scored as a miss. The fix was to make the eval check whether a chunk's
slide *range* **covers** the expected slide number, not just whether it
equals it — a good example of a bug that was in the measurement, not the
thing being measured.

**Phase 3 — Grounded chat.** Coming soon.

**Phase 4 — Modes (Open / Deep).** Coming soon.

**Phase 5 — Multi-notebook.** Coming soon.

**Phase 6 — Audio overview.** Coming soon.

**Phase 7 — UI.** Coming soon.

**Phase 8 — Extras.** Coming soon.
