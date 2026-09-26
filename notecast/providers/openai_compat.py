"""An adapter that looks like `anthropic.Anthropic` (duck-typed) to the rest
of the codebase, but talks to AgentAUS — Trellis Data's OpenAI-compatible
`/v1/chat/completions` API — via the `openai` Python SDK.

The call sites in `notecast/chat` and `notecast/audio` were written against
Claude-specific request/response shapes (search_result content blocks,
citations, `output_config`, `messages.stream(...).get_final_message()`,
`messages.parse(...)`, ...). Rather than rewriting every call site, this
module translates those Anthropic-style kwargs into OpenAI chat.completions
requests, and translates the OpenAI response back into small objects that
expose the same attributes call sites already read (`.content`,
`.stop_reason`, `.usage.input_tokens`, `.parsed_output`, ...).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import openai

from notecast.chat.client import ChatError
from notecast.config import Settings

# --- Response shapes (duck-typed Anthropic Message) -------------------


@dataclass
class SearchResultCitation:
    type: str = "search_result_location"
    search_result_index: int = 0
    cited_text: str = ""
    source: str | None = None
    title: str | None = None


@dataclass
class TextBlock:
    type: str = "text"
    text: str = ""
    citations: list[SearchResultCitation] | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    server_tool_use: Any = None


@dataclass
class Message:
    id: str = ""
    type: str = "message"
    role: str = "assistant"
    model: str = ""
    content: list[TextBlock] = field(default_factory=list)
    stop_reason: str | None = "end_turn"
    stop_details: Any = None
    usage: Usage = field(default_factory=Usage)
    parsed_output: Any = None


# --- Small helpers -------------------------------------------------------


def _bget(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from an SDK object or a plain dict, defensively."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


_CITATION_INSTRUCTION = (
    "When you cite the sources given above, put the source's id in square brackets right "
    "after the claim it supports, like [3] or [2][5]. Only cite ids that actually appear in "
    "the sources given to you — never invent an id. Do not add a references/bibliography list "
    "at the end of your answer. Always put code, array indexing and formulas that use "
    "square brackets inside `backticks` so they are never mistaken for citations."
)

_WEB_SEARCH_NOTE = (
    "Web search is unavailable in this deployment. Answer from the course sources given to "
    "you and, where needed, your own general knowledge — clearly label anything from general "
    "knowledge as such (it cannot be cited to a source)."
)


def _flatten_system(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts: list[str] = []
    for block in system:
        if _bget(block, "type", "text") == "text":
            text = _bget(block, "text", "") or ""
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _render_source_block(source_id: int, title: str, source: str, content_text: str) -> str:
    return f'<source id="{source_id}" title="{title}" file="{source}">\n{content_text}\n</source>'


def _flatten_block(block: Any, counter: list[int], sources: list[dict[str, Any]]) -> str:
    if isinstance(block, str):
        return block
    block_type = _bget(block, "type")
    if block_type == "text":
        return _bget(block, "text", "") or ""
    if block_type == "search_result":
        counter[0] += 1
        source_id = counter[0]
        title = _bget(block, "title") or ""
        source = _bget(block, "source") or ""
        content_blocks = _bget(block, "content") or []
        text_parts = [
            _bget(cb, "text", "") or "" for cb in content_blocks if _bget(cb, "text", None)
        ]
        content_text = "\n".join(text_parts)
        sources.append({"id": source_id, "title": title, "source": source, "content": content_text})
        return _render_source_block(source_id, title, source, content_text)
    # Unknown block type (e.g. a tool_use/tool_result echoed back in
    # assistant content): use its text if it has one, else drop it.
    text = _bget(block, "text", None)
    return text or ""


def _flatten_content(content: Any, counter: list[int], sources: list[dict[str, Any]]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = [_flatten_block(block, counter, sources) for block in content]
    return "\n\n".join(part for part in parts if part)


def _build_request(
    settings: Settings, kwargs: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], bool, type | None]:
    """Translate Anthropic-style `messages.create`/`.stream`/`.parse` kwargs
    into an OpenAI `chat.completions.create` request.

    Returns (request, sources, json_requested, parse_model).
    """
    model = kwargs.get("model")
    max_tokens = kwargs.get("max_tokens")
    if max_tokens is not None:
        max_tokens = min(int(max_tokens), settings.agentaus_max_output_tokens)

    system_text = _flatten_system(kwargs.get("system"))

    counter = [0]
    sources: list[dict[str, Any]] = []
    openai_messages: list[dict[str, str]] = []
    for message in kwargs.get("messages", []) or []:
        role = _bget(message, "role")
        content = _bget(message, "content")
        openai_messages.append(
            {"role": role, "content": _flatten_content(content, counter, sources)}
        )

    extra_system_parts: list[str] = []
    if sources:
        extra_system_parts.append(_CITATION_INSTRUCTION)

    tools = kwargs.get("tools")
    if tools:
        names = {_bget(tool, "name") for tool in tools}
        if "web_search" in names:
            extra_system_parts.append(_WEB_SEARCH_NOTE)

    parse_model = kwargs.get("output_format")
    schema: dict[str, Any] | None = None
    output_config = kwargs.get("output_config")
    fmt = output_config.get("format") if isinstance(output_config, dict) else None
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        schema = fmt.get("schema")
    elif parse_model is not None:
        schema = parse_model.model_json_schema()

    json_requested = schema is not None
    if json_requested:
        extra_system_parts.append(
            "Respond with ONLY a single JSON object matching this JSON Schema, no prose, "
            "no code fences:\n" + json.dumps(schema)
        )

    if extra_system_parts:
        system_text = "\n\n".join(part for part in [system_text, *extra_system_parts] if part)

    final_messages: list[dict[str, str]] = []
    if system_text:
        final_messages.append({"role": "system", "content": system_text})
    final_messages.extend(openai_messages)

    request: dict[str, Any] = {"model": model, "messages": final_messages}
    if max_tokens is not None:
        request["max_tokens"] = max_tokens

    return request, sources, json_requested, parse_model


# --- Response translation: think-stripping, code fences, citations -----

_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _strip_think(text: str) -> str:
    """Strip a complete leading `<think>...</think>` block. An unterminated
    leading `<think>` (no closing tag) is left untouched.
    """
    return _THINK_RE.sub("", text, count=1)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
    return stripped


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json(text: str) -> Any | None:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    candidate = _first_balanced_object(cleaned)
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# A single bracketed citation group: [3], [2, 5], [2,5], [2-4].
_MARKER_GROUP_RE = re.compile(r"\[\s*\d+(?:\s*[,-]\s*\d+)*\s*\]")


def _parse_marker_ids(group_text: str) -> list[int]:
    inner = group_text.strip()[1:-1]
    ids: list[int] = []
    for part in inner.split(","):
        part = part.strip()
        if "-" in part:
            start_s, _, end_s = part.partition("-")
            try:
                start_i, end_i = int(start_s.strip()), int(end_s.strip())
            except ValueError:
                continue
            ids.extend(range(start_i, end_i + 1))
        else:
            try:
                ids.append(int(part))
            except ValueError:
                continue
    return ids


_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*?`")


def _code_spans(text: str) -> list[tuple[int, int]]:
    """Byte ranges of fenced (```...```) and inline (`...`) code spans, so
    bracket text inside them is never mistaken for a citation marker.

    Fenced spans are masked out (replaced with a same-length placeholder
    that has no backticks) before scanning for inline spans, so a stray
    backtick from a `` ``` `` fence — or unpaired backticks inside fenced
    content — can never make the inline regex swallow real text between,
    before, or after a fenced block.
    """
    spans = [(m.start(), m.end()) for m in _FENCED_CODE_RE.finditer(text)]
    masked = text
    for start, end in spans:
        masked = masked[:start] + ("\x00" * (end - start)) + masked[end:]
    for m in _INLINE_CODE_RE.finditer(masked):
        spans.append((m.start(), m.end()))
    return spans


def _in_code_span(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(span_start <= start < span_end for span_start, span_end in spans)


# Continuation chars that, right after a bracket's "]", read as code rather
# than prose: an assignment/index/call, an arithmetic operator, another
# identifier character, or a "." immediately followed by a letter/digit
# (attribute/method access) — handled separately below.
_CODE_CONTINUATION_RE = re.compile(r"[A-Za-z0-9_=\[(+\-*/]")


def _looks_like_code_index(text: str, start: int, end: int) -> bool:
    """True when a bracketed `[...]` run at `text[start:end]` reads as code
    indexing (`x[2]=`, `matrix[1][2]`, `foo()[1].bar`, `arr[1]+1`) rather
    than a citation marker.

    False negatives on prose are worse than rare false positives here (code
    is already protected separately by `_code_spans`), so this only flags a
    run as code when BOTH sides look like code: preceded by an identifier
    char or a closing `)`/`]`, AND followed by a code continuation. A bare
    prose case like "the set[1]" or a leading "[1] According to..." is left
    as a citation.
    """
    prev_char = text[start - 1] if start > 0 else ""
    if not prev_char or not re.match(r"[A-Za-z0-9_)\]]", prev_char):
        return False

    next_char = text[end] if end < len(text) else ""
    if not next_char:
        return False
    if next_char == ".":
        return end + 1 < len(text) and text[end + 1].isalnum()
    return bool(_CODE_CONTINUATION_RE.match(next_char))


def _find_marker_runs(text: str) -> list[tuple[int, int, list[int]]]:
    """Find runs of one or more adjacent bracket-groups (e.g. "[2][5]"),
    each as (start, end, ids). Runs inside code spans, or that read as code
    indexing rather than citations, are left out entirely (and so left
    untouched in the rendered text).
    """
    code_spans = _code_spans(text)
    matches = list(_MARKER_GROUP_RE.finditer(text))
    runs: list[tuple[int, int, list[int]]] = []
    i = 0
    while i < len(matches):
        start = matches[i].start()
        end = matches[i].end()
        # The code-index check looks at the character right after the FIRST
        # group's closing bracket, so chained indexing like "matrix[1][2]"
        # (next char "[") is still caught once groups are merged below.
        first_group_end = end
        ids = _parse_marker_ids(matches[i].group(0))
        j = i + 1
        while j < len(matches) and text[end : matches[j].start()].strip() == "":
            end = matches[j].end()
            ids.extend(_parse_marker_ids(matches[j].group(0)))
            j += 1
        if not _in_code_span(start, end, code_spans) and not _looks_like_code_index(
            text, start, first_group_end
        ):
            runs.append((start, end, ids))
        i = j
    return runs


_CITED_TEXT_MAX_LEN = 300


def _split_citations(text: str, sources: list[dict[str, Any]]) -> list[TextBlock]:
    sources_by_id = {source["id"]: source for source in sources}
    runs = _find_marker_runs(text)
    if not runs:
        return [TextBlock(text=text, citations=None)]

    blocks: list[TextBlock] = []
    pos = 0
    for start, end, ids in runs:
        preceding = text[pos:start]
        seen: set[int] = set()
        valid_ids: list[int] = []
        for source_id in ids:
            if source_id in sources_by_id and source_id not in seen:
                seen.add(source_id)
                valid_ids.append(source_id)
        citations: list[SearchResultCitation] | None = None
        if valid_ids:
            citations = [
                SearchResultCitation(
                    search_result_index=source_id - 1,
                    cited_text=sources_by_id[source_id]["content"][:_CITED_TEXT_MAX_LEN],
                    source=sources_by_id[source_id].get("source"),
                    title=sources_by_id[source_id].get("title"),
                )
                for source_id in valid_ids
            ]
        blocks.append(TextBlock(text=preceding, citations=citations))
        pos = end

    trailing = text[pos:]
    if trailing:
        blocks.append(TextBlock(text=trailing, citations=None))
    return blocks


_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
    "tool_calls": "tool_use",
}


def _map_finish_reason(reason: str | None) -> str:
    return _FINISH_REASON_MAP.get(reason or "", "end_turn")


def _cache_read_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    details = _bget(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    return _bget(details, "cached_tokens", 0) or 0


def _translate_response(
    completion: Any,
    *,
    sources: list[dict[str, Any]],
    json_requested: bool,
    parse_model: type | None,
) -> Message:
    choice = completion.choices[0]
    message = choice.message
    raw_text = _bget(message, "content", "") or ""
    text = _strip_think(raw_text)

    parsed_output: Any = None
    if json_requested:
        clean_text = _strip_code_fences(text)
        content = [TextBlock(text=clean_text, citations=None)]
        if parse_model is not None:
            json_obj = _extract_json(text)
            if json_obj is not None:
                try:
                    parsed_output = parse_model.model_validate(json_obj)
                except Exception:
                    parsed_output = None
    elif sources:
        content = _split_citations(text, sources)
    else:
        content = [TextBlock(text=text, citations=None)]

    usage = getattr(completion, "usage", None)
    usage_obj = Usage(
        input_tokens=(_bget(usage, "prompt_tokens", 0) or 0) if usage is not None else 0,
        output_tokens=(_bget(usage, "completion_tokens", 0) or 0) if usage is not None else 0,
        cache_read_input_tokens=_cache_read_tokens(usage),
        cache_creation_input_tokens=0,
        server_tool_use=None,
    )

    return Message(
        id=getattr(completion, "id", "") or "",
        model=getattr(completion, "model", "") or "",
        content=content,
        stop_reason=_map_finish_reason(getattr(choice, "finish_reason", None)),
        stop_details=None,
        usage=usage_obj,
        parsed_output=parsed_output,
    )


# --- Error mapping --------------------------------------------------------


def map_openai_error(exc: Exception) -> ChatError:
    """Map an `openai` SDK exception to a friendly, AgentAUS-specific
    `ChatError`, checking the most specific exception types first.
    """
    if isinstance(exc, openai.AuthenticationError):
        return ChatError("AgentAUS rejected the API key. Check AGENTAUS_API_KEY in your .env file.")
    if isinstance(exc, openai.NotFoundError):
        return ChatError(
            "AgentAUS API returned 'not found' — check NOTECAST_AGENTAUS_MODEL and "
            "AGENTAUS_BASE_URL (it should end in /v1) in your .env file."
        )
    if isinstance(exc, openai.RateLimitError):
        retry_after = None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = headers.get("retry-after")
        message = "You've hit the AgentAUS API rate limit."
        if retry_after:
            message += f" Retry after {retry_after}s."
        return ChatError(message)
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code >= 500:
            return ChatError("AgentAUS's API had a server error. Please try again shortly.")
        return ChatError(f"AgentAUS API request failed: {exc.message}")
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return ChatError(
            "Could not connect to AgentAUS. Check AGENTAUS_BASE_URL and your network connection."
        )
    if isinstance(exc, openai.OpenAIError):
        return ChatError(f"AgentAUS API error: {exc}")
    return ChatError(f"Unexpected error calling the AgentAUS API: {exc}")


# --- The adapter itself ---------------------------------------------------


class _MessagesNamespace:
    """Duck-types `anthropic.Anthropic().messages`."""

    def __init__(self, outer: OpenAICompatClient) -> None:
        self._outer = outer

    def create(self, **kwargs: Any) -> Message:
        return self._outer._complete(**kwargs)

    def parse(self, **kwargs: Any) -> Message:
        return self._outer._complete(**kwargs)

    def stream(self, **kwargs: Any) -> _StreamContext:
        return _StreamContext(self._outer, kwargs)


class _StreamContext:
    """Duck-types the context manager `anthropic`'s `messages.stream(...)`
    returns. AgentAUS calls are done non-streaming internally; this just
    defers the request until `get_final_message()` is called, inside the
    `with` block, matching call sites' usage.
    """

    def __init__(self, outer: OpenAICompatClient, kwargs: dict[str, Any]) -> None:
        self._outer = outer
        self._kwargs = kwargs

    def __enter__(self) -> _StreamContext:
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def get_final_message(self) -> Message:
        return self._outer._complete(**self._kwargs)


class OpenAICompatClient:
    """Looks like `anthropic.Anthropic` (duck-typed) to call sites, but
    talks to AgentAUS's OpenAI-compatible `/v1/chat/completions` API via the
    `openai` Python SDK.
    """

    def __init__(self, settings: Settings, *, openai_client: Any | None = None) -> None:
        self._settings = settings
        self._client = openai_client or openai.OpenAI(
            api_key=settings.agentaus_api_key,
            base_url=settings.agentaus_base_url,
            timeout=settings.agentaus_timeout_seconds,
            max_retries=2,
        )
        # Once the server 400s on `response_format`, stop sending it for the
        # rest of this client's life instead of failing every request twice.
        self._skip_response_format = False
        self.messages = _MessagesNamespace(self)

    def list_models(self) -> list[str]:
        try:
            response = self._client.models.list()
        except openai.OpenAIError as exc:
            raise map_openai_error(exc) from exc
        return [model.id for model in response]

    def _complete(self, **kwargs: Any) -> Message:
        request, sources, json_requested, parse_model = _build_request(self._settings, kwargs)

        send_response_format = (
            json_requested and self._settings.agentaus_json_mode and not self._skip_response_format
        )
        if send_response_format:
            request["response_format"] = {"type": "json_object"}

        try:
            completion = self._client.chat.completions.create(**request)
        except openai.BadRequestError as exc:
            if send_response_format:
                self._skip_response_format = True
                retry_request = dict(request)
                retry_request.pop("response_format", None)
                try:
                    completion = self._client.chat.completions.create(**retry_request)
                except openai.OpenAIError as retry_exc:
                    raise map_openai_error(retry_exc) from retry_exc
            else:
                raise map_openai_error(exc) from exc
        except openai.OpenAIError as exc:
            raise map_openai_error(exc) from exc

        return _translate_response(
            completion, sources=sources, json_requested=json_requested, parse_model=parse_model
        )
