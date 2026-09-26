# Using NoteCast with AgentAUS

This branch adds **AgentAUS**, a sovereign Australian model from Trellis
Data, as an alternative to Claude. AgentAUS exposes an **OpenAI-compatible
API** (the same shape Open Notebook uses for it), so NoteCast talks to it
through the `openai` SDK behind an adapter that looks just like the Claude
client to the rest of the app. On this branch, AgentAUS is the **default
provider**. Claude is still fully supported — see
[Switching back to Claude](#switching-back-to-claude) below.

This page assumes you've already read the main setup guide
([`docs/SETUP-WINDOWS.md`](SETUP-WINDOWS.md)) and
[`docs/USAGE.md`](USAGE.md). It only covers what's different with AgentAUS.

## What stays the same, what changes

Embeddings/retrieval (`fastembed`, local) and audio TTS (Kokoro, local) are
**unchanged** — they never call any LLM. With `NOTECAST_PROVIDER=agentaus`,
the only network call that leaves your machine for chat and audio scripting
goes to AgentAUS itself, at `AGENTAUS_BASE_URL`. NoteCast makes no claims
about where that endpoint is hosted or what sovereignty guarantees it
carries — that's between you and Trellis Data.

## Setup

1. **Get an AgentAUS API key.** This is the same key you'd use to configure
   Open Notebook's OpenAI-compatible provider for AgentAUS. If you don't
   have one yet, ask whoever issued your AgentAUS access.

2. **Add it to `.env`:**

   ```bash
   AGENTAUS_API_KEY=...
   ```

   That's it — `NOTECAST_PROVIDER=agentaus`, `AGENTAUS_BASE_URL`
   (`https://agentaus.com.au/api/v1`) and `NOTECAST_AGENTAUS_MODEL`
   (`agentaus.v1`) already default to the real AgentAUS endpoint (see
   `notecast/config.py`), so you only need to set the key unless you're
   pointing at something else. Note the `/api` in the base URL — plain
   `/v1` redirects to a login page rather than serving the API.

3. **Check the connection:**

   ```bash
   uv run notecast provider-check
   ```

   This prints the active provider and the models NoteCast will use for
   each role (chat, helper, script, deep), lists the models the endpoint
   itself offers, and sends one small "ping" request to confirm the key,
   base URL and model ID actually work together. Run this before anything
   else — it's the fastest way to catch a bad key or a typo in an override.

4. **Use NoteCast as usual** — `ingest`, `ask`, `chat`, `audio` all work the
   same way regardless of provider. See [`docs/USAGE.md`](USAGE.md) for the
   full command reference.

## How the adapter works

`notecast/providers/openai_compat.py` provides `OpenAICompatClient`, which
presents the same interface the rest of NoteCast expects from the Claude
client, so retrieval, chat, and audio scripting code doesn't need to know
which provider is behind it. Under the hood:

- **Citations.** Claude has native citations; AgentAUS doesn't. Instead, the
  adapter numbers each retrieved source in the prompt as `<source id="n">`,
  asks the model to mark what it uses with `[n]`, and converts those
  markers back into NoteCast's usual citations (file + page/slide/
  timestamp) so the citation chips in the UI work the same way. The one
  difference: the cited span is the whole source excerpt the chunk came
  from, not the exact sentence Claude would point at, so citations are
  slightly less precise.
- **Open mode.** Course sources plus the model's own general knowledge,
  clearly labelled — same as Claude's open mode, minus a web search *tool
  of NoteCast's own*. AgentAUS has its own built-in web search that can
  switch itself on uninvited, even when NoteCast requests `tool_choice:
  none` — see [Troubleshooting](#troubleshooting) below.
- **System prompt.** Every call sends `system_prompt_overwrite: true`
  (`NOTECAST_AGENTAUS_SYSTEM_PROMPT_OVERWRITE`, default on), which replaces
  Trellis's hidden default system instructions — about 2,400 tokens — with
  NoteCast's own, much shorter prompt. Without it, every call would carry
  that hidden overhead in addition to NoteCast's own instructions; with it,
  a typical call's system prompt is roughly 130 tokens.
- **Structured output** (audio key points, chapter scripts, the coverage
  check) is requested by describing the JSON Schema in the prompt, plus
  `response_format={"type": "json_object"}` when
  `NOTECAST_AGENTAUS_JSON_MODE=true` (the default). AgentAUS accepts that
  parameter but doesn't actually enforce it — replies aren't guaranteed to
  be valid JSON just because you asked. NoteCast's real safety net is that
  it validates every JSON reply itself and retries once with an explicit
  "JSON only" instruction if parsing fails; the `response_format` fallback
  (dropping the parameter if a server rejects it outright) is still there
  too, it just isn't the main mechanism for AgentAUS.
- **Reasoning.** AgentAUS's reasoning is hidden — no `<think>` tags have
  been observed in practice — but it still counts toward output tokens
  (roughly 70 tokens of hidden reasoning even for a two-word reply, which
  is part of why replies can feel slow for their length). On the rare
  occasions something leaks through, the adapter strips leading plain-text
  reasoning lines (things like "We should use web search.") and raw
  control tokens (a `<|start|>...` style sequence, cut at the first `<|`).

## Differences vs Claude

| | Claude (Anthropic) | AgentAUS |
|---|---|---|
| Citations | Native, points at exact cited sentence | Emulated via `[n]` markers; cites the whole source excerpt |
| Web search (open mode) | Server-side web search tool, NoteCast-controlled | No tool of NoteCast's own; AgentAUS's own built-in search can switch on uninvited (see Troubleshooting) |
| Prompt caching | Yes — deep mode follow-ups are cheap | No — deep mode resends the full scope every turn |
| System prompt overhead | N/A | Replaced via `system_prompt_overwrite` — otherwise a hidden ~2,400-token default is added to every call |
| Context window (input + output) | Up to ~1M tokens | 131,072 tokens; over that, HTTP 400 (`max_model_len 131072`), shown as a friendly error |
| Output length | Config default, respected | `max_tokens` is ignored — the model self-limits to roughly 2,500 words per reply (~100 tokens/sec; longest observed ~6.7k tokens including hidden reasoning) |
| Helper/cheap model | Haiku 4.5 for helper calls | None — `/models` lists exactly one model, `agentaus.v1`; the server accepts any model name you send and just replies as "agentaus" |
| Cost estimates | Built-in pricing table | AgentAUS pricing isn't published; leave the price settings unset and NoteCast shows `n/a` instead of a number |

The practical effect: deep mode and audio generation cost more per call with
AgentAUS (no caching), and deep mode has less room for "whole course at
once" than Claude's roughly 1M-token window, though 131k tokens is still
generous for a course of the size NoteCast has been tested on. Output is the
other thing to watch — NoteCast's audio chapters (around 850 words each, up
to 8 chapters for a 45-minute episode) and its compact merge-and-rank format
for AgentAUS are both sized to fit under the ~2,500-word self-limit;
stretching `NOTECAST_AUDIO_MAX_MINUTES` far past 45 could push individual
chapters past what a single reply can hold.

## Tuning knobs

All of these live in `.env`, read with an `NOTECAST_` prefix (except the
API key and base URL, which are also read as plain `AGENTAUS_API_KEY` /
`AGENTAUS_BASE_URL`):

- `AGENTAUS_BASE_URL` (default `https://agentaus.com.au/api/v1`) — only
  change this if you're pointing at a different AgentAUS deployment.
- `NOTECAST_AGENTAUS_MODEL` (default `agentaus.v1`) — the model used for
  every role (chat, script, deep) unless that role has an explicit
  override. `/models` currently lists only this one model.
- `NOTECAST_AGENTAUS_HELPER_MODEL` — optional cheaper/faster model for query
  rewriting and audio key-point extraction. No such model exists on
  AgentAUS today, so this falls back to `NOTECAST_AGENTAUS_MODEL`.
- `NOTECAST_CHAT_MODEL` / `NOTECAST_HELPER_MODEL` / `NOTECAST_SCRIPT_MODEL` /
  `NOTECAST_DEEP_MODEL` — per-role overrides. If you set any of these
  explicitly, they win over the AgentAUS defaults above.
- `NOTECAST_AGENTAUS_CONTEXT_TOKENS` (default 131072) — deep mode refuses a
  scope that won't fit in this many tokens (input + output). This matches
  AgentAUS's actual context window; deep mode budgets down to roughly 119k
  tokens of material after leaving room for the question and the reply.
- `NOTECAST_AGENTAUS_MAX_OUTPUT_TOKENS` (default 8192) — the value NoteCast
  sends and budgets around, even though AgentAUS ignores `max_tokens` in
  practice and self-limits to about 2,500 words per reply either way.
- `NOTECAST_AGENTAUS_JSON_MODE` (default true) — sends
  `response_format={"type": "json_object"}` for structured requests.
  AgentAUS accepts this but doesn't enforce it, so NoteCast validates and
  retries regardless; leave this on unless you find a reason not to.
- `NOTECAST_AGENTAUS_SYSTEM_PROMPT_OVERWRITE` (default true) — sends
  `system_prompt_overwrite: true` so NoteCast's own system prompt replaces
  Trellis's hidden default one. Leave this on; turning it off adds a large,
  irrelevant hidden prompt to every call for no benefit.
- `NOTECAST_AGENTAUS_TIMEOUT_SECONDS` (default 600) — request timeout.
- `NOTECAST_AGENTAUS_PRICE_INPUT_PER_MTOK` /
  `NOTECAST_AGENTAUS_PRICE_OUTPUT_PER_MTOK` — USD per million tokens. Set
  both to get cost estimates before deep-mode/audio calls, same as Claude's
  built-in pricing. AgentAUS doesn't publish prices, so these are unset by
  default and NoteCast shows `n/a` instead of a number.

## Troubleshooting

**401 / authentication error**
Check `AGENTAUS_API_KEY` in `.env`. Run `uv run notecast provider-check` to
confirm it's actually being read (it prints the active provider, not the
key itself).

**404, redirected to a login page, or "model not found"**
Almost always the base URL: it needs the `/api` in
`https://agentaus.com.au/api/v1` — a plain `/v1` redirects to AgentAUS's
login page instead of serving the API. If you've overridden
`NOTECAST_AGENTAUS_MODEL`, also check it matches what `/models` lists (just
`agentaus.v1` today). Run `uv run notecast provider-check` — it lists the
models the endpoint offers, so you can copy the exact ID.

**HTTP 400, `max_model_len 131072`**
Your request (material + question + expected reply) went over AgentAUS's
131,072-token context window. In deep mode, narrow the scope (fewer
weeks) — NoteCast budgets to roughly 119k tokens of material to leave room
for the rest, but a very large scope can still exceed it. This error can
also show up if AgentAUS's own web search kicks in unexpectedly and adds a
large chunk of extra input tokens to the call (see the web search note
below); retrying the same question sometimes succeeds once search doesn't
trigger.

**Answers or audio scripts look cut off**
AgentAUS ignores `max_tokens` and self-limits each reply to roughly 2,500
words regardless of `NOTECAST_AGENTAUS_MAX_OUTPUT_TOKENS`, so raising that
setting won't help. If you're hitting this in audio generation, it's
usually a single chapter or the merge-and-rank step running long — try a
narrower scope, or check whether `NOTECAST_AUDIO_MAX_MINUTES` has been
pushed well past the default 45.

**JSON errors while generating audio scripts / key points**
AgentAUS accepts `response_format={"type": "json_object"}` but doesn't
actually enforce valid JSON, so occasional malformed replies are expected.
NoteCast validates every JSON reply and retries once with an explicit "JSON
only" instruction, which resolves most of these automatically; if you're
still seeing failures after that retry, it's worth checking the reply
length isn't hitting the ~2,500-word self-limit mid-structure.
`--script-only` audio runs are the cheapest way to iterate on this without
paying for TTS rendering each time.

**Calls that are unusually slow, or unexpectedly hit the context limit**
AgentAUS has its own built-in web search that can switch itself on
uninvited — even when NoteCast sends `tool_choice: none` — and add
anywhere from 20k to 100k input tokens to a single call. This is a known
AgentAUS behaviour, not a NoteCast bug, and it's a common cause of both slow
individual calls and unexpected context-overflow errors in deep mode. There
is no NoteCast-side setting to disable it. NoteCast's own open mode still
has no web search tool of its own — this is purely something the AgentAUS
server does on its own initiative.

**Cost estimate shows "n/a"**
Expected — AgentAUS pricing isn't published, so
`NOTECAST_AGENTAUS_PRICE_INPUT_PER_MTOK` / `..._OUTPUT_PER_MTOK` are unset
by default. Set them yourself if you learn real numbers; otherwise this is
normal, not a bug.

## Switching back to Claude

Set `NOTECAST_PROVIDER=anthropic` in `.env` (and make sure
`ANTHROPIC_API_KEY` is set). Everything else — commands, notebooks, indexes
— is unaffected; only the LLM backend changes. `uv run notecast
provider-check` confirms which provider and models are active either way.
