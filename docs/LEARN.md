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

**Phase 3 — Grounded chat.** Built the Claude client, the sources-only
system prompt, citation mapping, and `notecast ask` / `notecast chat`.

*`search_result` blocks, and why they're better than "just ask Claude to
cite its sources".* You could just paste your chunks into the prompt as
plain text and ask Claude to write `[1]`, `[2]` markers, then try to match
those markers back to the right chunk yourself with string parsing. That's
fragile — the model can miscount, skip a marker, or cite something it
didn't actually use. NoteCast instead sends each retrieved chunk as a
`search_result` content block (a native Claude API feature), and Claude
returns **structured citations**: each part of its answer comes back tagged
with exactly which block (and which sentence-range inside it) it's based
on. There's no marker-parsing to get wrong, and every citation is
guaranteed to point at real content NoteCast actually sent.

*The sources-only contract.* In `sources` mode, the system prompt (see
`notecast/chat/prompts.py`) tells Claude, in plain terms: answer only from
what's in these search results, don't fill gaps from general knowledge, and
if the results don't actually answer the question, say so plainly instead
of guessing. To make that check reliable rather than hoping Claude phrases
it consistently, the prompt asks for an exact marker token,
`[NOT_IN_SOURCES]`, as the very first thing in the reply when this happens.
NoteCast looks for that literal token, strips it out, and shows you a
friendly "Not in your course material" instead — same idea as the
`[NOT_IN_SOURCES]` you'll see mentioned in `docs/USAGE.md`, just translated
into a message a student reads, not a tag the code reads.

*Open mode and web search.* `open` mode uses the same retrieval step, but
adds Claude's own general knowledge and a server-side `web_search` tool it
can call itself when your notes don't cover something. The system prompt
requires Claude to keep these three separate in its answer — course
material, the web, and its own uncited knowledge — so you always know how
much to trust a given sentence.

*Query rewriting for follow-ups.* Search works on a single query string,
but chat is multi-turn: "and what about the second one?" means nothing to a
search index on its own. Before every retrieval, a small, cheap Haiku 4.5
call rewrites the latest message into a standalone query using the recent
conversation as context ("what about the second one?" → "what is the second
smoothing technique mentioned"), without touching what actually gets shown
to you — only the search step sees the rewritten version.

*Prompt caching, and the bug where it made things worse, not better.*
**Prompt caching** lets you tell Claude's API "remember this exact block of
input for a few minutes so a later request that starts with the same bytes
doesn't have to pay full price for it again" — cache writes cost 1.25x the
normal input price, but a cache *read* only costs about a tenth of it. The
saving only shows up if the same content is actually resent unchanged on a
later call. The first version of chat caching got this backwards: it placed
the cache breakpoint on that turn's *search results* — which are different
almost every question, since retrieval returns whatever's most relevant to
that specific question. That meant NoteCast was paying the 25% write
premium on nearly every question and almost never getting a matching cache
read back, i.e. caching was making chat slightly *more* expensive than not
caching at all. The fix moved the cache breakpoint onto the **conversation
history** instead — the growing list of past questions and answers, which
*is* genuinely stable and byte-identical from one turn to the next within a
session. Now the part that repeats gets cached, and the part that changes
every time (this turn's search results) doesn't pay a write premium it'll
rarely recoup.

**Phase 4 — Modes (Open / Deep).** Built `deep` mode: full-course context
with caching, plus its pre-flight cost estimate.

*Why sending the whole course works here.* Top-k retrieval is great for a
focused question but can miss things for a question like "compare lecture 2
and lecture 5" or "list every assumption made this unit" — the right
answer needs pieces scattered across many chunks, more than any reasonable
top-k would fetch. Current Claude models have a **1-million-token context
window**, and this course's *entire* material is only around 47,000 tokens
— small enough to just send everything, every time, and let Claude read all
of it rather than guess which handful of chunks matter. That's specific to
a course being this size; it wouldn't make sense for a course ten times
larger, which is why deep mode is opt-in and shows a token/cost estimate
first (see `notecast/chat/session.py`'s `estimate_deep_cost`) rather than
being the default.

*Caching the material.* The whole-course material is sent once, as
`search_result` blocks, with a cache breakpoint on the end of it. The first
question in a scope pays the cache-write premium on all of it (≈$0.12 for
this course, on Sonnet 5); every follow-up question in the same scope reads
it back from cache instead (≈$0.01). If you change the scope (a different
`--week` filter, say), the material is different, so the cache doesn't
apply and NoteCast rebuilds and re-attaches it fresh.

*The citations-dropped bug the verifier caught.* Deep mode's citation
mapper needs the list of chunks that were actually sent, in order, to turn
each citation index Claude returns back into a real file/slide/timestamp.
The first version of deep mode called that mapper with an **empty** chunk
list instead of the ordered material it had just sent — so every citation
Claude returned was silently thrown away, and deep-mode answers came back
with no citations at all, no error, nothing visibly wrong unless you
noticed the citations were simply missing. The fix keeps the ordered chunk
list stored alongside the cached material blocks (rather than reconstructing
or forgetting it) and passes that same list to the citation mapper, so
citation indices resolve against the chunks Claude was actually shown.

**Phase 5 — Multi-notebook.** A "notebook" was always just a folder under
`notebooks/<slug>/` with its own `sources/`, its own LanceDB table, and its
own manifest — so "multi-notebook support" is mostly already true by
construction rather than a separate system to build. This phase added the
small pieces that make switching between courses convenient day to day:
`notecast notebooks create` / `notecast notebooks list` to manage them
without touching the filesystem by hand, and the `--week` / source-path
scope filters used throughout `search`, `ask`, `chat` and `audio` so a
question or an audio overview can be scoped to part of a course instead of
always meaning "the whole notebook".

**Phase 6 — Audio overview.** Built the key-point extraction, ranking,
adaptive-length planner, two-host scriptwriter, coverage check, and Kokoro
TTS rendering.

*Why coverage-first, not top-k, for a summary.* Retrieval answers "which
chunks are most *similar* to this query" — great for a question, wrong tool
for a summary, where the goal is the opposite: make sure nothing important
gets left out, even things that don't closely resemble each other. So the
audio pipeline is built around **coverage** end to end: extract every
concept worth knowing first, rank it, budget time for it, then check
afterwards that the ranked points actually made it into the script —
rather than trusting a single generation pass to remember everything.

*The five (really seven) steps.* 1) **Key points**: one cheap Haiku 4.5
call per source file lists every concept, definition, method, worked
example and caveat, each scored for importance (explicit lecturer emphasis,
time spent on it, appearing in *both* slides and transcript, assessment
mentions). 2) **Merge + rank**: one Sonnet 5 call dedupes those across all
sources and sorts them into Tier A (must cover), B (cover briefly), C
(mention or skip). 3) **Budget**: turn the tier counts into a time budget
(≈90s per A point, 30s per B, clamped 5–45 minutes) — compress B points
first if it'd run long, never drop an A point. 4) **Outline into chapters**.
5) **Script chapter by chapter**, each call given the full material plus a
running summary so two hosts (one explains, one asks the questions a
student would actually ask) stay consistent across the episode. 6)
**Coverage check**: a Haiku 4.5 pass verifies every Tier A point is
genuinely explained somewhere in the script, and regenerates any chapter
that missed one. 7) **TTS**: Kokoro renders each line locally and the
result is stitched into one MP3 with chapter markers.

*Structured outputs.* Both the key-point extraction and the merge/rank step
ask Claude for validated JSON (a fixed shape — a list of points with named
fields) rather than free-form prose to parse with regex, so the rest of the
pipeline can trust the shape of what comes back instead of guessing at it.

*Why a coverage check at all.* Even with a good outline, a single
generation pass can quietly skip something — the model runs out of room in
a chapter, or just doesn't get to a point it was supposed to cover. Treating
the script as "done" without checking would mean occasionally shipping an
episode that never actually explains something you were told is important.
The coverage check is a second, independent pass whose only job is to
verify that, so gaps get caught and fixed automatically instead of only
being caught by you noticing something's missing after listening.

*Bugs the verifier caught here:*
- **Truncated JSON.** A chapter script is written in one Claude call with a
  fixed `max_tokens` budget. When a chapter's script actually needed more
  room than that budget, the response got cut off mid-JSON and the whole
  call crashed trying to parse it, instead of failing gracefully or getting
  more room. The fix checks the response's `stop_reason`: on a genuine
  length cutoff, it retries once with a larger `max_tokens` and an added
  "be more concise" instruction, and it sizes the *initial* `max_tokens`
  from the chapter's own target word count in the first place (so most
  chapters never need the retry at all), with a friendly error if it still
  doesn't fit after retrying.
- **Silent coverage drop.** The coverage checker returns two lists —
  "covered" and "missing" tier-A point ids — but the first version only
  acted on what was in "missing", so a point the model simply left out of
  *both* lists (neither confirmed covered nor flagged missing) vanished
  from tracking entirely and never got fixed. The fix stopped trusting the
  model's own "missing" list at all: anything not explicitly confirmed
  present in "covered" now counts as missing, so a point can't slip through
  by being left out of both.
- **Week order.** Course order (for both the audio outline and deep mode's
  material) is supposed to be week 1, week 2, ... week 10, but the first
  version sorted chunks by source file *path* first, and a folder named
  `week-10/` sorts alphabetically before `week-2/` (because the character
  `"1"` is less than `"2"`), so week 10's material was read before week 2's.
  The fix sorts by the numeric week number first, and only falls back to
  file path as a tie-breaker within the same week.
- **Key-point merge retry format.** `merge_and_rank` normally asks the SDK
  to validate its JSON reply against a pydantic model directly. Its retry
  path (used when the first attempt's output didn't parse) instead built
  the expected reply format by hand, without `additionalProperties: false`,
  which let the model's retry response drift from the shape the rest of the
  pipeline expected. The fix has the retry go through the same path as the
  first attempt — passing the pydantic model itself to the SDK's
  `messages.stream(output_format=...)` — so both attempts get the same
  strict, validated schema instead of the retry using a looser hand-built
  one.

**Phase 7 — UI.** Built the Streamlit web UI: notebook picker, chat, and
audio generation, on top of the same core the CLI uses.

*Streamlit reruns, in plain terms.* Streamlit's programming model is
unusual if you're used to a normal app: it doesn't have persistent "event
handlers" — instead, the **entire page script re-runs top to bottom** every
time you interact with anything (click a button, type in a box, change a
dropdown). State that should survive between reruns (which notebook is
selected, the chat history so far, a cached retriever) has to be explicitly
stashed in `st.session_state`; anything else resets on every interaction.

*Why paid actions sit behind buttons and confirmations.* Because the whole
script reruns on almost any interaction, if generating an audio overview or
asking a deep-mode question just happened as a normal part of the script
running top to bottom, it would risk firing again on some unrelated rerun
(you resize a widget, you type another character somewhere else) — an
accidental extra charge to your API key for something you didn't actually
ask for again. The UI avoids this the same way the CLI's `--yes` flag does:
anything that spends API money only happens inside a button's own click
handler (which only runs on that specific click, not on every rerun), and
for deep mode / audio generation it shows the same cost estimate and asks
for confirmation first, exactly like the CLI does.

**Phase 8 — Extras (ideas).**
- Quiz / flashcards / study-guide generation from a notebook's material,
  using the same coverage-first approach as audio (make sure every
  important point gets a question, not just whatever the model thinks of
  first).
- Optional Whisper transcription (for lectures you only have as audio/video,
  not already-transcribed) and OCR (for scanned/handwritten slides) — both
  dropped from the original scope to keep Phase 1 simple, revisit later.
- Try the optional reranker (already wired up, off by default) against a
  second real course to see if it's worth turning on generally.
