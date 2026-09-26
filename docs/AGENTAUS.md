# Using NoteCast with AgentAUS

This branch adds **AgentAUS**, a sovereign Australian model from Trellis
Data, as an alternative to Claude. AgentAUS exposes an **OpenAI-compatible
`/v1` API** (the same shape Open Notebook uses for it), so NoteCast talks to
it through the `openai` SDK behind an adapter that looks just like the
Claude client to the rest of the app. On this branch, AgentAUS is the
**default provider**. Claude is still fully supported — see
[Switching back to Claude](#switching-back-to-claude) below.

This page assumes you've already read the main setup guide
([`docs/SETUP-WINDOWS.md`](SETUP-WINDOWS.md)) and
[`docs/USAGE.md`](USAGE.md). It only covers what's different with AgentAUS.

## What stays the same, what changes

Embeddings/retrieval (`fastembed`, local) and audio TTS (Kokoro, local) are
**unchanged** — they never call any LLM. With `NOTECAST_PROVIDER=agentaus`,
the only network call that leaves your machine for chat and audio scripting
goes to AgentAUS itself, at whatever `AGENTAUS_BASE_URL` you configure.
NoteCast makes no claims about where that endpoint is hosted or what
sovereignty guarantees it carries — that's between you and Trellis Data.

## Setup

1. **Get an AgentAUS API key and endpoint.** These are the same base URL and
   key you'd have used to configure Open Notebook's OpenAI-compatible
   provider for AgentAUS. If you don't have them yet, ask whoever issued
   your AgentAUS access.

2. **Add them to `.env`:**

   ```bash
   NOTECAST_PROVIDER=agentaus
   AGENTAUS_API_KEY=...
   AGENTAUS_BASE_URL=https://<your-agentaus-endpoint>/v1
   NOTECAST_AGENTAUS_MODEL=<model id>
   ```

   The base URL must include the `/v1` suffix, same as you'd have used in
   Open Notebook. The model ID is whatever Open Notebook listed for this
   endpoint — if you're not sure, the next step confirms it.

3. **Check the connection:**

   ```bash
   uv run notecast provider-check
   ```

   This prints the active provider and the models NoteCast will use for
   each role (chat, helper, script, deep), lists the models the endpoint
   itself offers, and sends one small "ping" request to confirm the key,
   base URL and model ID actually work together. Run this before anything
   else — it's the fastest way to catch a typo in the URL or model ID.

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
  clearly labelled — same as Claude's open mode, minus the web search tool.
  AgentAUS doesn't have a server-side web search tool, so open mode with
  AgentAUS never fetches anything from the web.
- **Structured output** (audio key points, chapter scripts, the coverage
  check) is requested by describing the JSON Schema in the prompt, plus
  `response_format={"type": "json_object"}` when
  `NOTECAST_AGENTAUS_JSON_MODE=true` (the default). If the endpoint rejects
  that parameter, the adapter automatically retries without it.
- **Reasoning traces.** Some AgentAUS responses include `<think>...</think>`
  blocks. The adapter strips these before parsing or displaying output.

## Differences vs Claude

| | Claude (Anthropic) | AgentAUS |
|---|---|---|
| Citations | Native, points at exact cited sentence | Emulated via `[n]` markers; cites the whole source excerpt |
| Web search (open mode) | Server-side web search tool | Not available — open mode is sources + model knowledge only |
| Prompt caching | Yes — deep mode follow-ups are cheap | No — deep mode resends the full scope every turn |
| Deep mode context limit | Up to ~1M tokens | Capped at `NOTECAST_AGENTAUS_CONTEXT_TOKENS` (default 128,000) |
| Max output per request | Config default | Capped at `NOTECAST_AGENTAUS_MAX_OUTPUT_TOKENS` (default 8192) |
| Cost estimates | Built-in pricing table | Only shown if you set `NOTECAST_AGENTAUS_PRICE_INPUT_PER_MTOK` / `..._OUTPUT_PER_MTOK` |

The practical effect: deep mode and audio generation cost more per call with
AgentAUS (no caching), and deep mode has less room for "whole course at
once" on courses bigger than the context cap. Sources-only and open-mode
chat are the least affected, since those are already small requests.

## Tuning knobs

All of these live in `.env`, read with an `NOTECAST_` prefix (except the
API key and base URL, which are also read as plain `AGENTAUS_API_KEY` /
`AGENTAUS_BASE_URL`):

- `NOTECAST_AGENTAUS_MODEL` — the model used for every role (chat, script,
  deep) unless that role has an explicit override.
- `NOTECAST_AGENTAUS_HELPER_MODEL` — optional cheaper/faster model for query
  rewriting and audio key-point extraction. Falls back to
  `NOTECAST_AGENTAUS_MODEL` if unset.
- `NOTECAST_CHAT_MODEL` / `NOTECAST_HELPER_MODEL` / `NOTECAST_SCRIPT_MODEL` /
  `NOTECAST_DEEP_MODEL` — per-role overrides. If you set any of these
  explicitly, they win over the AgentAUS defaults above.
- `NOTECAST_AGENTAUS_CONTEXT_TOKENS` (default 128000) — deep mode refuses a
  scope that won't fit in this many tokens. Raise it if your AgentAUS model
  supports a bigger context window; lower it if you're hitting context
  errors from the server.
- `NOTECAST_AGENTAUS_MAX_OUTPUT_TOKENS` (default 8192) — cap applied to
  every request's max output tokens. Raise it if the model supports more
  and you're seeing truncated answers or scripts.
- `NOTECAST_AGENTAUS_JSON_MODE` (default true) — sends
  `response_format={"type": "json_object"}` for structured requests. Set to
  `false` if your endpoint rejects that parameter (see Troubleshooting).
- `NOTECAST_AGENTAUS_TIMEOUT_SECONDS` (default 600) — request timeout.
- `NOTECAST_AGENTAUS_PRICE_INPUT_PER_MTOK` /
  `NOTECAST_AGENTAUS_PRICE_OUTPUT_PER_MTOK` — USD per million tokens. Set
  both to get cost estimates before deep-mode/audio calls, same as Claude's
  built-in pricing. Leave unset if you don't know AgentAUS pricing yet —
  NoteCast just won't show an estimate.

## Troubleshooting

**401 / authentication error**
Check `AGENTAUS_API_KEY` in `.env`. Run `uv run notecast provider-check` to
confirm it's actually being read (it prints the active provider, not the
key itself).

**404, or "model not found"**
Usually one of two things: `AGENTAUS_BASE_URL` is missing the `/v1` suffix,
or `NOTECAST_AGENTAUS_MODEL` doesn't match an ID the endpoint actually
serves. Run `uv run notecast provider-check` — it lists the models the
endpoint offers, so you can copy the exact ID.

**`response_format` rejected by the server**
Some OpenAI-compatible servers don't support
`response_format={"type": "json_object"}`. The adapter retries without it
automatically, but if you see repeated JSON errors, set
`NOTECAST_AGENTAUS_JSON_MODE=false` in `.env` to skip straight to
prompt-only JSON requests.

**Deep mode refuses a scope ("too large for context")**
Your material for that scope exceeds `NOTECAST_AGENTAUS_CONTEXT_TOKENS`
(default 128,000). Narrow the scope (fewer weeks), or raise the setting if
your AgentAUS model actually supports a bigger window — check with whoever
manages the endpoint first, since raising it past what the model supports
will just fail differently.

**Answers or audio scripts look cut off**
Raise `NOTECAST_AGENTAUS_MAX_OUTPUT_TOKENS`, if the model allows more output
per request. The default (8192) is conservative.

**JSON errors while generating audio scripts / key points**
This is usually the model producing near-valid-but-not-quite JSON despite
the schema in the prompt. Try `NOTECAST_AGENTAUS_JSON_MODE=true` if it was
off, or use a more capable model for `NOTECAST_AGENTAUS_HELPER_MODEL` /
`NOTECAST_SCRIPT_MODEL`. `--script-only` audio runs are the cheapest way to
iterate on this without paying for TTS rendering each time.

## Switching back to Claude

Set `NOTECAST_PROVIDER=anthropic` in `.env` (and make sure
`ANTHROPIC_API_KEY` is set). Everything else — commands, notebooks, indexes
— is unaffected; only the LLM backend changes. `uv run notecast
provider-check` confirms which provider and models are active either way.
