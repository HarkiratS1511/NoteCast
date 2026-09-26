"""Tests for notecast.chat.session.ChatSession, with a fully fake Claude
client and retriever — no network access.
"""

from __future__ import annotations

from typing import Any

import pytest

from notecast.chat.client import ChatError
from notecast.chat.session import ChatSession
from notecast.config import Settings
from notecast.models import Chunk, Location, SearchHit, SourceType

# --- Fakes -------------------------------------------------------------


class FakeStreamContext:
    def __init__(self, message: Any) -> None:
        self._message = message

    def __enter__(self) -> FakeStreamContext:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def get_final_message(self) -> Any:
        return self._message


class FakeMessages:
    def __init__(
        self,
        stream_responses: list[Any] | None = None,
        create_response: Any = None,
    ) -> None:
        self._stream_responses = list(stream_responses or [])
        self._create_response = create_response
        self.stream_calls: list[dict] = []
        self.create_calls: list[dict] = []

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        if isinstance(self._create_response, Exception):
            raise self._create_response
        return self._create_response

    def stream(self, **kwargs: Any) -> FakeStreamContext:
        self.stream_calls.append(kwargs)
        message = self._stream_responses.pop(0)
        if isinstance(message, Exception):
            raise message
        return FakeStreamContext(message)


class FakeClient:
    def __init__(self, stream_responses=None, create_response=None) -> None:
        self.messages = FakeMessages(stream_responses, create_response)


class FakeRetriever:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[dict] = []

    def search(self, query, k: int = 10, filters=None, mode: str = "hybrid"):
        self.calls.append({"query": query, "k": k, "filters": filters, "mode": mode})
        return self.hits


# --- Fixtures ------------------------------------------------------------


def _settings(**overrides: Any) -> Settings:
    defaults = dict(
        anthropic_api_key="sk-ant-test",
        chat_model="claude-sonnet-5",
        helper_model="claude-haiku-4-5",
        chat_effort="medium",
        chat_max_tokens=8000,
        web_search_max_uses=3,
        retrieval_top_k=5,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _hit(chunk_id: str = "c1") -> SearchHit:
    chunk = Chunk(
        chunk_id=chunk_id,
        course="comp4650",
        source_path="week-03/lecture-2.pptx",
        source_type=SourceType.PPTX,
        ordinal=0,
        text="Some slide content about graphs.",
        header="COMP4650 · Week 3 · slide 4",
        location=Location(slide=4),
        week=3,
    )
    return SearchHit(chunk=chunk, score=0.9)


def _end_turn_message(text: str = "The answer is X.", **usage_overrides: Any) -> dict:
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 5,
        "server_tool_use": {"web_search_requests": 0},
    }
    usage.update(usage_overrides)
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": usage,
    }


# --- Tests -----------------------------------------------------------------


def test_first_turn_has_no_rewrite_call() -> None:
    retriever = FakeRetriever([_hit()])
    client = FakeClient(stream_responses=[_end_turn_message()])
    session = ChatSession(retriever, settings=_settings(), client=client)

    answer = session.ask("What is a graph traversal?")

    assert answer.standalone_query == "What is a graph traversal?"
    assert client.messages.create_calls == []
    assert retriever.calls[0]["query"] == "What is a graph traversal?"


def test_follow_up_triggers_rewrite_with_helper_model() -> None:
    retriever = FakeRetriever([_hit()])
    client = FakeClient(
        stream_responses=[_end_turn_message()],
        create_response={"content": [{"type": "text", "text": "standalone rewritten query"}]},
    )
    session = ChatSession(retriever, settings=_settings(), client=client)
    session.history.append(("earlier question", "earlier answer"))

    answer = session.ask("what about the second one?")

    assert len(client.messages.create_calls) == 1
    assert client.messages.create_calls[0]["model"] == "claude-haiku-4-5"
    assert answer.standalone_query == "standalone rewritten query"
    assert retriever.calls[0]["query"] == "standalone rewritten query"


def test_rewrite_failure_falls_back_to_raw_question() -> None:
    retriever = FakeRetriever([_hit()])
    client = FakeClient(
        stream_responses=[_end_turn_message()],
        create_response=RuntimeError("helper model unavailable"),
    )
    session = ChatSession(retriever, settings=_settings(), client=client)
    session.history.append(("earlier question", "earlier answer"))

    answer = session.ask("what about that?")

    assert answer.standalone_query == "what about that?"
    assert retriever.calls[0]["query"] == "what about that?"


def test_sources_mode_zero_hits_short_circuits_without_api_call() -> None:
    retriever = FakeRetriever([])
    client = FakeClient(stream_responses=[])
    session = ChatSession(retriever, settings=_settings(), client=client)

    answer = session.ask("something not covered", mode="sources")

    assert answer.not_in_sources is True
    assert client.messages.stream_calls == []
    assert answer.usage.input_tokens == 0


def test_open_mode_includes_web_search_tool_sources_mode_does_not() -> None:
    retriever = FakeRetriever([_hit()])

    client_open = FakeClient(stream_responses=[_end_turn_message()])
    session_open = ChatSession(retriever, settings=_settings(), client=client_open)
    session_open.ask("question", mode="open")
    open_kwargs = client_open.messages.stream_calls[0]
    assert open_kwargs["tools"][0]["type"] == "web_search_20260209"
    assert open_kwargs["tools"][0]["max_uses"] == 3

    retriever2 = FakeRetriever([_hit()])
    client_sources = FakeClient(stream_responses=[_end_turn_message()])
    session_sources = ChatSession(retriever2, settings=_settings(), client=client_sources)
    session_sources.ask("question", mode="sources")
    sources_kwargs = client_sources.messages.stream_calls[0]
    assert "tools" not in sources_kwargs


def test_request_uses_settings_models_effort_and_cache_control_no_temperature() -> None:
    retriever = FakeRetriever([_hit()])
    client = FakeClient(stream_responses=[_end_turn_message()])
    settings = _settings(chat_effort="high", chat_max_tokens=12000)
    session = ChatSession(retriever, settings=settings, client=client)

    session.ask("question")

    kwargs = client.messages.stream_calls[0]
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["max_tokens"] == 12000
    assert kwargs["output_config"] == {"effort": "high"}
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    # No assistant prefill: every message we send is user or assistant text,
    # and the outbound request always ends on a user turn.
    assert kwargs["messages"][-1]["role"] == "user"


def test_pause_turn_continues_and_accumulates_usage() -> None:
    retriever = FakeRetriever([_hit()])
    paused = {
        "content": [{"type": "text", "text": "Let me think..."}],
        "stop_reason": "pause_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "server_tool_use": {"web_search_requests": 1},
        },
    }
    final = _end_turn_message("Final answer.", input_tokens=5, output_tokens=15)
    client = FakeClient(stream_responses=[paused, final])
    session = ChatSession(retriever, settings=_settings(), client=client)

    answer = session.ask("question", mode="open")

    assert len(client.messages.stream_calls) == 2
    assert answer.stop_reason == "end_turn"
    assert answer.usage.input_tokens == 15
    assert answer.usage.output_tokens == 35
    assert answer.usage.web_searches == 1
    # The continuation re-sent the paused assistant turn.
    second_call_messages = client.messages.stream_calls[1]["messages"]
    assert second_call_messages[-1]["role"] == "assistant"


def test_refusal_raises_chat_error() -> None:
    retriever = FakeRetriever([_hit()])
    refusal_message = {
        "content": [],
        "stop_reason": "refusal",
        "stop_details": {"type": "refusal", "category": "cyber"},
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }
    client = FakeClient(stream_responses=[refusal_message])
    session = ChatSession(retriever, settings=_settings(), client=client)

    with pytest.raises(ChatError):
        session.ask("question")


def test_history_appended_without_citation_markers() -> None:
    retriever = FakeRetriever([_hit()])
    message = {
        "content": [
            {
                "type": "text",
                "text": "Graphs are traversed with BFS or DFS.",
                "citations": [
                    {
                        "type": "search_result_location",
                        "cited_text": "Some slide content about graphs.",
                        "source": "week-03/lecture-2.pptx",
                        "title": "COMP4650 · Week 3 · slide 4",
                        "search_result_index": 0,
                        "start_block_index": 0,
                        "end_block_index": 1,
                    }
                ],
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 10,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }
    client = FakeClient(stream_responses=[message])
    session = ChatSession(retriever, settings=_settings(), client=client)

    answer = session.ask("How do you traverse a graph?")

    assert answer.text == "Graphs are traversed with BFS or DFS.[1]"
    assert session.history == [
        ("How do you traverse a graph?", "Graphs are traversed with BFS or DFS.")
    ]


def test_cost_estimate_is_populated_for_known_model() -> None:
    retriever = FakeRetriever([_hit()])
    client = FakeClient(stream_responses=[_end_turn_message()])
    session = ChatSession(retriever, settings=_settings(), client=client)

    answer = session.ask("question")

    assert answer.usage.est_cost_usd is not None
    assert answer.usage.est_cost_usd > 0


def test_reset_clears_history() -> None:
    retriever = FakeRetriever([_hit()])
    client = FakeClient(stream_responses=[_end_turn_message()])
    session = ChatSession(retriever, settings=_settings(), client=client)
    session.ask("question")
    assert session.history

    session.reset()

    assert session.history == []
