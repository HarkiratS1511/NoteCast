"""Multi-turn grounded chat session: retrieve, ask Claude, parse citations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import anthropic

from notecast.chat.client import ChatError, get_client, map_anthropic_error
from notecast.chat.grounding import (
    chunks_to_search_results,
    hits_to_search_results,
    parse_response,
    sort_chunks_for_deep,
)
from notecast.chat.models import AnswerSegment, ChatAnswer, ChatMode, Usage
from notecast.chat.pricing import estimate_cost
from notecast.chat.prompts import (
    DEEP_SYSTEM_PROMPT,
    OPEN_SYSTEM_PROMPT,
    QUERY_REWRITE_PROMPT,
    SOURCES_ONLY_SYSTEM_PROMPT,
)
from notecast.config import Settings, get_settings
from notecast.ingest.textutils import estimate_tokens
from notecast.models import Chunk, SearchFilters, SearchHit

_MAX_PAUSE_CONTINUATIONS = 3
_QUERY_REWRITE_MAX_TOKENS = 200
# Deep mode: refuse to send material estimated over this many input tokens.
_MAX_DEEP_TOKENS = 600_000
# Rough per-chunk JSON overhead (title, source, block wrapper) added on top
# of the chunk's own text when estimating deep mode's material size.
_DEEP_OVERHEAD_TOKENS_PER_CHUNK = 20


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class ChatSession:
    """One ongoing grounded-chat conversation for a notebook."""

    def __init__(
        self,
        retriever: Any,
        *,
        settings: Settings | None = None,
        client: anthropic.Anthropic | None = None,
        course_name: str = "",
        chunk_source: Callable[[SearchFilters | None], list[Chunk]] | None = None,
    ) -> None:
        self.retriever = retriever
        self.settings = settings or get_settings()
        self.client = client or get_client(self.settings)
        self.course_name = course_name
        # Deep mode needs every chunk in scope, not just top-k hits; the
        # service layer passes e.g. `store.all_chunks`.
        self.chunk_source = chunk_source
        self.history: list[tuple[str, str]] = []
        self._last_mode: ChatMode | None = None
        # The deep-mode "material" context currently attached to a turn in
        # `self.history`: {"blocks": [...], "attach_at": <history index>}.
        # Rebuilt from scratch whenever the mode or the deep scope changes
        # (see `_ask_deep`).
        self._deep_material: dict[str, Any] | None = None
        self._deep_scope_key: tuple | None = None

    def reset(self) -> None:
        """Clear conversation history and any deep-mode material context."""
        self.history = []
        self._last_mode = None
        self._deep_material = None
        self._deep_scope_key = None

    def ask(
        self,
        question: str,
        *,
        mode: ChatMode = "sources",
        filters: SearchFilters | None = None,
        k: int | None = None,
    ) -> ChatAnswer:
        if not question or not question.strip():
            raise ValueError("Please type a question.")

        if mode == "deep":
            answer = self._ask_deep(question, filters)
            self._last_mode = mode
            return answer

        standalone_query = self._rewrite_query(question)
        k = k or self.settings.retrieval_top_k
        hits = self.retriever.search(standalone_query, k=k, filters=filters)

        if mode == "sources" and not hits:
            answer = self._not_in_sources_answer(question, standalone_query, mode)
            self.history.append((question, self._plain_text(answer)))
            self._last_mode = mode
            return answer

        search_result_blocks = hits_to_search_results(hits)
        user_content: list[dict] = [*search_result_blocks, {"type": "text", "text": question}]

        messages: list[dict] = [
            *self._history_messages(),
            {"role": "user", "content": user_content},
        ]

        system_prompt = SOURCES_ONLY_SYSTEM_PROMPT if mode == "sources" else OPEN_SYSTEM_PROMPT
        tools = None
        if mode == "open":
            tools = [
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": self.settings.web_search_max_uses,
                }
            ]

        message, usage_totals = self._call_claude(system_prompt, messages, tools)

        segments, citations, not_in_sources = parse_response(message, hits)

        usage = Usage(**usage_totals)
        usage.est_cost_usd = estimate_cost(self.settings.chat_model, usage_totals)

        answer = ChatAnswer(
            question=question,
            standalone_query=standalone_query,
            mode=mode,
            segments=segments,
            citations=citations,
            not_in_sources=not_in_sources,
            hits=hits,
            usage=usage,
            model=self.settings.chat_model,
            stop_reason=_get(message, "stop_reason"),
        )
        self.history.append((question, self._plain_text(answer)))
        self._last_mode = mode
        return answer

    # --- Deep mode -----------------------------------------------------

    @staticmethod
    def _filters_key(filters: SearchFilters | None) -> tuple:
        """A hashable, order-independent signature for `filters`, so two
        equivalent filter objects compare equal.
        """
        if filters is None:
            return (None, None, None)
        weeks = tuple(sorted(filters.weeks)) if filters.weeks else None
        source_types = (
            tuple(sorted(t.value for t in filters.source_types)) if filters.source_types else None
        )
        source_paths = tuple(sorted(filters.source_paths)) if filters.source_paths else None
        return (weeks, source_types, source_paths)

    def _estimate_material_tokens(self, chunks: list[Chunk]) -> int:
        """Rough input-token estimate for sending `chunks` as deep mode's
        search_result material (chunk text plus JSON/header overhead).
        """
        total = 0
        for chunk in chunks:
            total += estimate_tokens(chunk.text) + estimate_tokens(chunk.header or "")
            total += _DEEP_OVERHEAD_TOKENS_PER_CHUNK
        return total

    def _deep_scope_chunks(self, filters: SearchFilters | None) -> tuple[list[Chunk], int]:
        """Fetch and order the chunks in scope for deep mode, and estimate
        their token size. Raises ChatError if there's no chunk source, or
        the scope is too large to send.
        """
        if self.chunk_source is None:
            raise ChatError("Deep mode needs the full index")
        chunks = self.chunk_source(filters)
        ordered = sort_chunks_for_deep(chunks)
        if not ordered:
            raise ChatError("Nothing in scope for deep mode — check the week filter")
        tokens = self._estimate_material_tokens(ordered)
        if tokens > _MAX_DEEP_TOKENS:
            raise ChatError(
                f"The material in scope is about {tokens:,} tokens, over deep mode's "
                f"{_MAX_DEEP_TOKENS:,}-token limit. Narrow the scope (fewer weeks or "
                "files) and try again."
            )
        return ordered, tokens

    def estimate_deep_cost(self, filters: SearchFilters | None = None) -> tuple[int, float]:
        """Estimate deep mode's material size and typical cost, so a caller
        can show it before the user confirms sending it.

        Returns (estimated_input_tokens, estimated_usd) where the USD figure
        covers a first, uncached call (which pays the cache-write premium on
        the material) plus one typical cached follow-up (which reads it back
        cheaply). Raises ChatError if there's no chunk source, or the scope
        is too large for deep mode.
        """
        _, tokens = self._deep_scope_chunks(filters)
        model = self.settings.deep_model
        first_call = estimate_cost(model, {"cache_write_tokens": tokens}) or 0.0
        follow_up = estimate_cost(model, {"cache_read_tokens": tokens}) or 0.0
        return tokens, first_call + follow_up

    def _ask_deep(self, question: str, filters: SearchFilters | None) -> ChatAnswer:
        """Deep mode: send every chunk in scope as search_result material,
        instead of top-k retrieval.

        The material is attached once, to whichever turn starts (or
        restarts) the current "deep scope" — the first deep turn after a
        mode switch or a change of `filters` — and resent identically on
        every later turn in that same scope so it reads from cache. If the
        mode or the scope changes, the material context is rebuilt fresh
        and reattached to the new turn; earlier turns keep appearing as
        plain-text history only (no material re-sent for them).

        The ordered chunk list is kept alongside the material blocks so
        citations can be mapped back: since only the current scope's
        material is ever in the request, `search_result_index` 0..N-1
        always lines up with that same chunk list, in the same order.
        """
        scope_key = self._filters_key(filters)
        needs_fresh_material = (
            self._deep_material is None
            or self._last_mode != "deep"
            or self._deep_scope_key != scope_key
        )
        if needs_fresh_material:
            ordered_chunks, _tokens = self._deep_scope_chunks(filters)
            material_blocks = chunks_to_search_results(ordered_chunks)
            if material_blocks:
                material_blocks[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            self._deep_material = {
                "blocks": material_blocks,
                "chunks": ordered_chunks,
                "attach_at": len(self.history),
            }
            self._deep_scope_key = scope_key

        attach_at = self._deep_material["attach_at"]
        material_blocks = self._deep_material["blocks"]
        material_hits = [
            SearchHit(chunk=chunk, score=1.0) for chunk in self._deep_material["chunks"]
        ]

        messages: list[dict] = []
        for i, (prior_question, prior_answer) in enumerate(self.history):
            if i == attach_at:
                user_content: list[dict] = [
                    *material_blocks,
                    {"type": "text", "text": prior_question},
                ]
            else:
                user_content = [{"type": "text", "text": prior_question}]
            messages.append({"role": "user", "content": user_content})
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": prior_answer}]}
            )
        if messages:
            messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        if attach_at == len(self.history):
            new_content: list[dict] = [*material_blocks, {"type": "text", "text": question}]
        else:
            new_content = [{"type": "text", "text": question}]
        messages.append({"role": "user", "content": new_content})

        message, usage_totals = self._call_claude(
            DEEP_SYSTEM_PROMPT, messages, tools=None, model=self.settings.deep_model
        )

        segments, citations, not_in_sources = parse_response(message, material_hits)

        usage = Usage(**usage_totals)
        usage.est_cost_usd = estimate_cost(self.settings.deep_model, usage_totals)

        answer = ChatAnswer(
            question=question,
            standalone_query=question,
            mode="deep",
            segments=segments,
            citations=citations,
            not_in_sources=not_in_sources,
            hits=material_hits,
            usage=usage,
            model=self.settings.deep_model,
            stop_reason=_get(message, "stop_reason"),
        )
        self.history.append((question, self._plain_text(answer)))
        return answer

    @staticmethod
    def _plain_text(answer: ChatAnswer) -> str:
        """The answer text without citation markers, for chat history."""
        return "".join(segment.text for segment in answer.segments)

    def _history_messages(self) -> list[dict]:
        """Prior turns as user/assistant text-block messages, with a cache
        breakpoint on the last block of the most recent turn (the previous
        assistant answer). History is append-only plain text, so this
        prefix is byte-identical across requests and reliably caches.
        """
        messages: list[dict] = []
        for prior_question, prior_answer in self.history:
            messages.append({"role": "user", "content": [{"type": "text", "text": prior_question}]})
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": prior_answer}]}
            )
        if messages:
            messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        return messages

    def _not_in_sources_answer(
        self, question: str, standalone_query: str, mode: ChatMode
    ) -> ChatAnswer:
        text = "The course material doesn't cover this — no relevant search results were found."
        return ChatAnswer(
            question=question,
            standalone_query=standalone_query,
            mode=mode,
            segments=[AnswerSegment(text=text, citation_numbers=[])],
            citations=[],
            not_in_sources=True,
            hits=[],
            usage=Usage(),
            model=self.settings.chat_model,
            stop_reason="end_turn",
        )

    def _rewrite_query(self, question: str) -> str:
        """Turn a follow-up question into a standalone search query using
        recent history, via the (cheap) helper model. Falls back to the raw
        question on any error, including on the first turn (no history).
        """
        if not self.history:
            return question
        try:
            recent = self.history[-3:]
            history_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in recent)
            user_content = f"Conversation history:\n{history_text}\n\nFollow-up: {question}"
            message = self.client.messages.create(
                model=self.settings.helper_model,
                max_tokens=_QUERY_REWRITE_MAX_TOKENS,
                system=QUERY_REWRITE_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            text = "".join(
                _get(block, "text", "") or ""
                for block in _get(message, "content", []) or []
                if _get(block, "type") == "text"
            ).strip()
            return text or question
        except Exception:
            return question

    def _call_claude(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        model: str | None = None,
    ) -> tuple[Any, dict]:
        """Call Claude, following `pause_turn` continuations (up to a limit)
        and mapping stop reasons. Returns (final_message, usage_totals).
        """
        kwargs: dict[str, Any] = {
            "model": model or self.settings.chat_model,
            "max_tokens": self.settings.chat_max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "output_config": {"effort": self.settings.chat_effort},
        }
        if tools:
            kwargs["tools"] = tools

        usage_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "web_searches": 0,
        }
        current_messages = list(messages)
        message = None
        continuations = 0
        while True:
            message = self._stream_once(current_messages, **kwargs)
            self._accumulate_usage(usage_totals, message)

            stop_reason = _get(message, "stop_reason")
            if stop_reason == "refusal":
                raise ChatError(self._refusal_message(message))
            if stop_reason == "pause_turn" and continuations < _MAX_PAUSE_CONTINUATIONS:
                continuations += 1
                current_messages = [
                    *current_messages,
                    {"role": "assistant", "content": _get(message, "content", [])},
                ]
                continue
            break

        return message, usage_totals

    def _stream_once(self, messages: list[dict], **kwargs: Any) -> Any:
        """One streaming request. Kept as its own method so tests can fake
        `self.client.messages.stream` without dealing with continuations.
        """
        try:
            with self.client.messages.stream(messages=messages, **kwargs) as stream:
                return stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise map_anthropic_error(exc) from exc

    @staticmethod
    def _refusal_message(message: Any) -> str:
        stop_details = _get(message, "stop_details")
        category = _get(stop_details, "category") if stop_details else None
        if category:
            return f"Claude declined to answer this question ({category})."
        return "Claude declined to answer this question."

    @staticmethod
    def _accumulate_usage(totals: dict, message: Any) -> None:
        usage = _get(message, "usage")
        if usage is None:
            return
        totals["input_tokens"] += _get(usage, "input_tokens", 0) or 0
        totals["output_tokens"] += _get(usage, "output_tokens", 0) or 0
        totals["cache_read_tokens"] += _get(usage, "cache_read_input_tokens", 0) or 0
        totals["cache_write_tokens"] += _get(usage, "cache_creation_input_tokens", 0) or 0
        server_tool_use = _get(usage, "server_tool_use")
        if server_tool_use is not None:
            totals["web_searches"] += _get(server_tool_use, "web_search_requests", 0) or 0
