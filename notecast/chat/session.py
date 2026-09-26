"""Multi-turn grounded chat session: retrieve, ask Claude, parse citations."""

from __future__ import annotations

from typing import Any

import anthropic

from notecast.chat.client import ChatError, get_client, map_anthropic_error
from notecast.chat.grounding import hits_to_search_results, parse_response
from notecast.chat.models import AnswerSegment, ChatAnswer, ChatMode, Usage
from notecast.chat.pricing import estimate_cost
from notecast.chat.prompts import (
    OPEN_SYSTEM_PROMPT,
    QUERY_REWRITE_PROMPT,
    SOURCES_ONLY_SYSTEM_PROMPT,
)
from notecast.config import Settings, get_settings
from notecast.models import SearchFilters

_MAX_PAUSE_CONTINUATIONS = 3
_QUERY_REWRITE_MAX_TOKENS = 200


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
    ) -> None:
        self.retriever = retriever
        self.settings = settings or get_settings()
        self.client = client or get_client(self.settings)
        self.course_name = course_name
        self.history: list[tuple[str, str]] = []

    def reset(self) -> None:
        """Clear conversation history."""
        self.history = []

    def ask(
        self,
        question: str,
        *,
        mode: ChatMode = "sources",
        filters: SearchFilters | None = None,
        k: int | None = None,
    ) -> ChatAnswer:
        standalone_query = self._rewrite_query(question)
        k = k or self.settings.retrieval_top_k
        hits = self.retriever.search(standalone_query, k=k, filters=filters)

        if mode == "sources" and not hits:
            answer = self._not_in_sources_answer(question, standalone_query, mode)
            self.history.append((question, self._plain_text(answer)))
            return answer

        search_result_blocks = hits_to_search_results(hits)
        user_content: list[dict] = [*search_result_blocks, {"type": "text", "text": question}]

        messages: list[dict] = []
        for prior_question, prior_answer in self.history:
            messages.append({"role": "user", "content": prior_question})
            messages.append({"role": "assistant", "content": prior_answer})
        messages.append({"role": "user", "content": user_content})

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
        return answer

    @staticmethod
    def _plain_text(answer: ChatAnswer) -> str:
        """The answer text without citation markers, for chat history."""
        return "".join(segment.text for segment in answer.segments)

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
        self, system_prompt: str, messages: list[dict], tools: list[dict] | None
    ) -> tuple[Any, dict]:
        """Call Claude, following `pause_turn` continuations (up to a limit)
        and mapping stop reasons. Returns (final_message, usage_totals).
        """
        kwargs: dict[str, Any] = {
            "model": self.settings.chat_model,
            "max_tokens": self.settings.chat_max_tokens,
            "system": system_prompt,
            "output_config": {"effort": self.settings.chat_effort},
            "cache_control": {"type": "ephemeral"},
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
