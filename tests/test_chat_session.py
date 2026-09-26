"""Tests for notecast.chat.session.ChatSession, with a fully fake Claude
client and retriever — no network access.
"""

from __future__ import annotations

from typing import Any

import pytest

from notecast.chat.client import ChatError
from notecast.chat.session import ChatSession
from notecast.config import Settings
from notecast.models import Chunk, Location, SearchFilters, SearchHit, SourceType

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
        deep_model="claude-sonnet-5",
        chat_effort="medium",
        chat_max_tokens=8000,
        web_search_max_uses=3,
        retrieval_top_k=5,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class FakeChunkSource:
    """A fake `chunk_source` callable: records the filters it was called
    with and returns a fixed list of chunks regardless of them (tests that
    care about filter behaviour pass different chunk lists per filter).
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.calls: list[SearchFilters | None] = []

    def __call__(self, filters: SearchFilters | None) -> list[Chunk]:
        self.calls.append(filters)
        return self.chunks


def _deep_chunk(
    chunk_id: str,
    *,
    week: int,
    source_type: SourceType,
    source_path: str,
    ordinal: int = 0,
    text: str = "Some course material.",
) -> Chunk:
    location = Location(slide=1) if source_type == SourceType.PPTX else Location()
    return Chunk(
        chunk_id=chunk_id,
        course="comp4650",
        source_path=source_path,
        source_type=source_type,
        ordinal=ordinal,
        text=text,
        header=f"COMP4650 · Week {week} · {source_path}",
        location=location,
        week=week,
    )


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


def test_request_uses_settings_models_and_effort_no_temperature() -> None:
    retriever = FakeRetriever([_hit()])
    client = FakeClient(stream_responses=[_end_turn_message()])
    settings = _settings(chat_effort="high", chat_max_tokens=12000)
    session = ChatSession(retriever, settings=settings, client=client)

    session.ask("question")

    kwargs = client.messages.stream_calls[0]
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["max_tokens"] == 12000
    assert kwargs["output_config"] == {"effort": "high"}
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    # No assistant prefill: every message we send is user or assistant text,
    # and the outbound request always ends on a user turn.
    assert kwargs["messages"][-1]["role"] == "user"


def test_system_prompt_sent_as_cached_block_no_top_level_cache_control() -> None:
    retriever = FakeRetriever([_hit()])
    client = FakeClient(stream_responses=[_end_turn_message()])
    session = ChatSession(retriever, settings=_settings(), client=client)

    session.ask("question", mode="sources")

    kwargs = client.messages.stream_calls[0]
    assert "cache_control" not in kwargs
    assert kwargs["system"] == [
        {
            "type": "text",
            "text": kwargs["system"][0]["text"],
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_first_turn_has_no_history_cache_breakpoint() -> None:
    retriever = FakeRetriever([_hit()])
    client = FakeClient(stream_responses=[_end_turn_message()])
    session = ChatSession(retriever, settings=_settings(), client=client)

    session.ask("question")

    messages = client.messages.stream_calls[0]["messages"]
    # No history yet, so the only message is the new user turn — nothing
    # should carry a cache_control breakpoint.
    assert len(messages) == 1
    assert "cache_control" not in messages[0]["content"][-1]


def test_second_turn_cache_breakpoint_on_last_history_block_only() -> None:
    retriever = FakeRetriever([_hit()])
    client = FakeClient(stream_responses=[_end_turn_message()])
    session = ChatSession(retriever, settings=_settings(), client=client)
    session.history.append(("earlier question", "earlier answer"))

    session.ask("question")

    messages = client.messages.stream_calls[0]["messages"]
    # [user(history q), assistant(history a, cached), user(new turn)]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert "cache_control" not in messages[0]["content"][-1]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # The new turn's search_result/question blocks must not be cached.
    new_turn_content = messages[2]["content"]
    assert all("cache_control" not in block for block in new_turn_content)


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


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_raises_value_error(question: str) -> None:
    retriever = FakeRetriever([_hit()])
    client = FakeClient(stream_responses=[])
    session = ChatSession(retriever, settings=_settings(), client=client)

    with pytest.raises(ValueError, match="Please type a question."):
        session.ask(question)

    assert retriever.calls == []
    assert client.messages.stream_calls == []


def test_reset_clears_history() -> None:
    retriever = FakeRetriever([_hit()])
    client = FakeClient(stream_responses=[_end_turn_message()])
    session = ChatSession(retriever, settings=_settings(), client=client)
    session.ask("question")
    assert session.history

    session.reset()

    assert session.history == []


# --- Deep mode ---------------------------------------------------------


def test_deep_sends_all_in_scope_chunks_in_course_order() -> None:
    # Deliberately out of order: week 5 before week 3, transcript before
    # slide within a week — deep mode must reorder them.
    chunks = [
        _deep_chunk("w5-slide", week=5, source_type=SourceType.PPTX, source_path="w5/a.pptx"),
        _deep_chunk("w3-transcript", week=3, source_type=SourceType.VTT, source_path="w3/a.vtt"),
        _deep_chunk("w3-slide", week=3, source_type=SourceType.PPTX, source_path="w3/a.pptx"),
    ]
    chunk_source = FakeChunkSource(chunks)
    retriever = FakeRetriever([])
    client = FakeClient(stream_responses=[_end_turn_message()])
    session = ChatSession(retriever, settings=_settings(), client=client, chunk_source=chunk_source)

    session.ask("Summarise the course.", mode="deep")

    kwargs = client.messages.stream_calls[0]
    material = kwargs["messages"][0]["content"][:-1]  # everything but the question block
    sources_in_order = [block["source"] for block in material]
    assert sources_in_order == ["w3/a.pptx", "w3/a.vtt", "w5/a.pptx"]
    # Deep mode doesn't retrieve top-k at all.
    assert retriever.calls == []


def test_deep_uses_deep_model_and_sources_only_style_prompt_no_tools() -> None:
    chunks = [_deep_chunk("c1", week=1, source_type=SourceType.PPTX, source_path="w1/a.pptx")]
    chunk_source = FakeChunkSource(chunks)
    client = FakeClient(stream_responses=[_end_turn_message()])
    settings = _settings(deep_model="claude-opus-5")
    session = ChatSession(
        FakeRetriever([]), settings=settings, client=client, chunk_source=chunk_source
    )

    answer = session.ask("Compare weeks.", mode="deep")

    kwargs = client.messages.stream_calls[0]
    assert kwargs["model"] == "claude-opus-5"
    assert "tools" not in kwargs
    assert "complete" in kwargs["system"][0]["text"].lower() or (
        "entire" in kwargs["system"][0]["text"].lower()
    )
    assert answer.model == "claude-opus-5"


def test_deep_turn_one_has_cache_control_on_last_material_block() -> None:
    chunks = [
        _deep_chunk("c1", week=1, source_type=SourceType.PPTX, source_path="w1/a.pptx"),
        _deep_chunk("c2", week=1, source_type=SourceType.PPTX, source_path="w1/b.pptx"),
    ]
    chunk_source = FakeChunkSource(chunks)
    client = FakeClient(stream_responses=[_end_turn_message()])
    session = ChatSession(
        FakeRetriever([]), settings=_settings(), client=client, chunk_source=chunk_source
    )

    session.ask("question", mode="deep")

    kwargs = client.messages.stream_calls[0]
    messages = kwargs["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    material_blocks = content[:-1]
    question_block = content[-1]
    assert question_block == {"type": "text", "text": "question"}
    # Only the very last material block carries the breakpoint.
    for block in material_blocks[:-1]:
        assert "cache_control" not in block["content"][-1]
    assert material_blocks[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_deep_turn_two_resends_identical_material_then_new_question() -> None:
    chunks = [_deep_chunk("c1", week=1, source_type=SourceType.PPTX, source_path="w1/a.pptx")]
    chunk_source = FakeChunkSource(chunks)
    client = FakeClient(
        stream_responses=[_end_turn_message("First answer."), _end_turn_message("Second answer.")]
    )
    session = ChatSession(
        FakeRetriever([]), settings=_settings(), client=client, chunk_source=chunk_source
    )

    session.ask("first question", mode="deep")
    session.ask("second question", mode="deep")

    assert len(client.messages.stream_calls) == 2
    turn1_messages = client.messages.stream_calls[0]["messages"]
    turn2_messages = client.messages.stream_calls[1]["messages"]

    # Turn 1: a single user message = [material..., question].
    turn1_material = turn1_messages[0]["content"][:-1]

    # Turn 2: [user(material..., first question), assistant(first answer,
    # cached), user(second question)].
    assert len(turn2_messages) == 3
    turn2_user_content = turn2_messages[0]["content"]
    turn2_material = turn2_user_content[:-1]
    turn2_question_block = turn2_user_content[-1]

    assert turn2_material == turn1_material
    assert turn2_question_block == {"type": "text", "text": "first question"}
    # The material's breakpoint is still there.
    assert turn2_material[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # The history breakpoint rule still applies to the most recent turn.
    assert turn2_messages[1]["role"] == "assistant"
    assert turn2_messages[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # The new question is appended on its own, with no material duplicated.
    assert turn2_messages[2] == {
        "role": "user",
        "content": [{"type": "text", "text": "second question"}],
    }

    # Only one chunk_source call was made — material was reused, not rebuilt.
    assert len(chunk_source.calls) == 1


def test_deep_respects_filters_passed_to_chunk_source() -> None:
    chunks = [_deep_chunk("c1", week=3, source_type=SourceType.PPTX, source_path="w3/a.pptx")]
    chunk_source = FakeChunkSource(chunks)
    client = FakeClient(stream_responses=[_end_turn_message()])
    session = ChatSession(
        FakeRetriever([]), settings=_settings(), client=client, chunk_source=chunk_source
    )
    filters = SearchFilters(weeks=[3])

    session.ask("question", mode="deep", filters=filters)

    assert chunk_source.calls == [filters]


def test_deep_without_chunk_source_raises_chat_error() -> None:
    client = FakeClient(stream_responses=[])
    session = ChatSession(FakeRetriever([]), settings=_settings(), client=client)

    with pytest.raises(ChatError, match="Deep mode needs the full index"):
        session.ask("question", mode="deep")

    assert client.messages.stream_calls == []


def test_deep_empty_scope_raises_chat_error_before_api_call() -> None:
    chunk_source = FakeChunkSource([])
    client = FakeClient(stream_responses=[])
    session = ChatSession(
        FakeRetriever([]), settings=_settings(), client=client, chunk_source=chunk_source
    )

    with pytest.raises(ChatError, match="Nothing in scope for deep mode — check the week filter"):
        session.ask("question", mode="deep")

    assert client.messages.stream_calls == []


def test_estimate_deep_cost_empty_scope_raises() -> None:
    chunk_source = FakeChunkSource([])
    session = ChatSession(
        FakeRetriever([]), settings=_settings(), client=FakeClient(), chunk_source=chunk_source
    )

    with pytest.raises(ChatError, match="Nothing in scope for deep mode"):
        session.estimate_deep_cost()


def test_estimate_deep_cost_returns_tokens_and_usd() -> None:
    chunks = [
        _deep_chunk(
            "c1", week=1, source_type=SourceType.PPTX, source_path="w1/a.pptx", text="x" * 4000
        )
    ]
    chunk_source = FakeChunkSource(chunks)
    session = ChatSession(
        FakeRetriever([]),
        settings=_settings(deep_model="claude-sonnet-5"),
        client=FakeClient(),
        chunk_source=chunk_source,
    )

    tokens, cost = session.estimate_deep_cost()

    assert tokens > 0
    assert cost > 0
    # A first, uncached call plus a cheap cached follow-up should cost less
    # than sending the material twice at full price.
    prices = 2.00  # claude-sonnet-5 input $/MTok, see notecast/chat/pricing.py
    full_price_twice = 2 * tokens / 1_000_000 * prices
    assert cost < full_price_twice


def test_estimate_deep_cost_without_chunk_source_raises() -> None:
    session = ChatSession(FakeRetriever([]), settings=_settings(), client=FakeClient())

    with pytest.raises(ChatError, match="Deep mode needs the full index"):
        session.estimate_deep_cost()


def test_deep_refuses_when_estimated_tokens_too_large() -> None:
    # ~4 chars/token estimate: 3M characters -> ~750k tokens, over the 600k limit.
    huge_chunk = _deep_chunk(
        "huge", week=1, source_type=SourceType.PPTX, source_path="w1/a.pptx", text="x" * 3_000_000
    )
    chunk_source = FakeChunkSource([huge_chunk])
    client = FakeClient(stream_responses=[])
    session = ChatSession(
        FakeRetriever([]), settings=_settings(), client=client, chunk_source=chunk_source
    )

    with pytest.raises(ChatError, match="600,000-token limit|600000"):
        session.ask("question", mode="deep")

    assert client.messages.stream_calls == []


def test_deep_estimate_cost_also_refuses_when_too_large() -> None:
    huge_chunk = _deep_chunk(
        "huge", week=1, source_type=SourceType.PPTX, source_path="w1/a.pptx", text="x" * 3_000_000
    )
    chunk_source = FakeChunkSource([huge_chunk])
    session = ChatSession(
        FakeRetriever([]), settings=_settings(), client=FakeClient(), chunk_source=chunk_source
    )

    with pytest.raises(ChatError):
        session.estimate_deep_cost()


def test_switching_from_sources_to_deep_starts_fresh_material_keeps_history() -> None:
    chunks = [_deep_chunk("c1", week=1, source_type=SourceType.PPTX, source_path="w1/a.pptx")]
    chunk_source = FakeChunkSource(chunks)
    client = FakeClient(
        stream_responses=[_end_turn_message("Sources answer."), _end_turn_message("Deep answer.")]
    )
    session = ChatSession(
        FakeRetriever([_hit()]), settings=_settings(), client=client, chunk_source=chunk_source
    )

    session.ask("sources question", mode="sources")
    session.ask("deep question", mode="deep")

    deep_messages = client.messages.stream_calls[1]["messages"]
    # Prior (sources-mode) turn appears as plain text only, no search_result
    # blocks re-sent for it.
    assert deep_messages[0] == {
        "role": "user",
        "content": [{"type": "text", "text": "sources question"}],
    }
    assert deep_messages[1]["role"] == "assistant"
    # The new deep turn carries the fresh material.
    deep_turn_content = deep_messages[2]["content"]
    assert deep_turn_content[-1] == {"type": "text", "text": "deep question"}
    assert deep_turn_content[0]["type"] == "search_result"


def test_sources_and_open_modes_still_work_unchanged() -> None:
    retriever = FakeRetriever([_hit()])
    client = FakeClient(stream_responses=[_end_turn_message(), _end_turn_message()])
    session = ChatSession(retriever, settings=_settings(), client=client)

    sources_answer = session.ask("question one", mode="sources")
    open_answer = session.ask("question two", mode="open")

    assert sources_answer.mode == "sources"
    assert open_answer.mode == "open"
    assert len(client.messages.stream_calls) == 2


# --- Deep mode citations (regression: search_result_index must map to the
# ordered material chunks actually in the request, not an empty hit list) ---


def _citing_message(text: str, *, search_result_index: int, cited_text: str) -> dict:
    return {
        "content": [
            {
                "type": "text",
                "text": text,
                "citations": [
                    {
                        "type": "search_result_location",
                        "cited_text": cited_text,
                        "source": "irrelevant",
                        "title": "irrelevant",
                        "search_result_index": search_result_index,
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


def test_deep_turn_one_citations_map_to_correct_chunk() -> None:
    # Sorted order (week 1, slide before transcript): chunk-a (pptx), then chunk-b (vtt).
    chunks = [
        _deep_chunk("chunk-a", week=1, source_type=SourceType.PPTX, source_path="w1/a.pptx"),
        _deep_chunk("chunk-b", week=1, source_type=SourceType.VTT, source_path="w1/a.vtt"),
    ]
    chunk_source = FakeChunkSource(chunks)
    message = _citing_message(
        "As covered in the slides.", search_result_index=0, cited_text="Some course material."
    )
    client = FakeClient(stream_responses=[message])
    session = ChatSession(
        FakeRetriever([]), settings=_settings(), client=client, chunk_source=chunk_source
    )

    answer = session.ask("question", mode="deep")

    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "chunk-a"
    assert answer.citations[0].source_path == "w1/a.pptx"
    assert answer.segments[0].citation_numbers == [1]
    assert answer.hits[0].chunk.chunk_id == "chunk-a"
    assert answer.hits[1].chunk.chunk_id == "chunk-b"


def test_deep_turn_two_same_scope_citations_still_map_correctly() -> None:
    chunks = [
        _deep_chunk("chunk-a", week=1, source_type=SourceType.PPTX, source_path="w1/a.pptx"),
        _deep_chunk("chunk-b", week=1, source_type=SourceType.VTT, source_path="w1/a.vtt"),
    ]
    chunk_source = FakeChunkSource(chunks)
    turn1 = _citing_message("First.", search_result_index=0, cited_text="Some course material.")
    turn2 = _citing_message("Second.", search_result_index=1, cited_text="Some course material.")
    client = FakeClient(stream_responses=[turn1, turn2])
    session = ChatSession(
        FakeRetriever([]), settings=_settings(), client=client, chunk_source=chunk_source
    )

    session.ask("first question", mode="deep")
    answer2 = session.ask("second question", mode="deep")

    assert len(answer2.citations) == 1
    assert answer2.citations[0].chunk_id == "chunk-b"
    assert answer2.citations[0].source_path == "w1/a.vtt"
    assert answer2.segments[0].citation_numbers == [1]
    # Material wasn't rebuilt for turn 2, but the hit list is still correct.
    assert len(chunk_source.calls) == 1


def test_switching_from_sources_to_deep_citations_map_to_deep_material() -> None:
    chunks = [
        _deep_chunk("chunk-a", week=1, source_type=SourceType.PPTX, source_path="w1/a.pptx"),
        _deep_chunk("chunk-b", week=1, source_type=SourceType.VTT, source_path="w1/a.vtt"),
    ]
    chunk_source = FakeChunkSource(chunks)
    sources_message = _end_turn_message("Sources answer.")
    deep_message = _citing_message(
        "Deep answer.", search_result_index=1, cited_text="Some course material."
    )
    client = FakeClient(stream_responses=[sources_message, deep_message])
    session = ChatSession(
        FakeRetriever([_hit()]), settings=_settings(), client=client, chunk_source=chunk_source
    )

    session.ask("sources question", mode="sources")
    deep_answer = session.ask("deep question", mode="deep")

    assert len(deep_answer.citations) == 1
    assert deep_answer.citations[0].chunk_id == "chunk-b"
    assert deep_answer.citations[0].source_path == "w1/a.vtt"


def _agentaus_settings(**overrides: Any) -> Settings:
    defaults = dict(
        _env_file=None,
        provider="agentaus",
        agentaus_api_key="key",
        agentaus_base_url="https://example.test/v1",
        agentaus_model="trellis-large",
        agentaus_context_tokens=128_000,
        agentaus_max_output_tokens=8192,
        chat_max_tokens=8000,
        deep_model="trellis-large",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_agentaus_max_deep_tokens_derived_from_context_window() -> None:
    chunk_source = FakeChunkSource([])
    session = ChatSession(
        FakeRetriever([]),
        settings=_agentaus_settings(),
        client=FakeClient(),
        chunk_source=chunk_source,
    )
    # 128_000 - min(8000, 8192) - 4_000 safety margin.
    assert session._max_deep_tokens() == 128_000 - 8_000 - 4_000


def test_agentaus_max_deep_tokens_uses_smaller_of_chat_and_output_cap() -> None:
    chunk_source = FakeChunkSource([])
    session = ChatSession(
        FakeRetriever([]),
        settings=_agentaus_settings(chat_max_tokens=20_000, agentaus_max_output_tokens=4_000),
        client=FakeClient(),
        chunk_source=chunk_source,
    )
    assert session._max_deep_tokens() == 128_000 - 4_000 - 4_000


def test_agentaus_max_deep_tokens_floors_at_small_positive_number() -> None:
    chunk_source = FakeChunkSource([])
    session = ChatSession(
        FakeRetriever([]),
        settings=_agentaus_settings(agentaus_context_tokens=1_000, agentaus_max_output_tokens=500),
        client=FakeClient(),
        chunk_source=chunk_source,
    )
    assert session._max_deep_tokens() > 0


def test_anthropic_max_deep_tokens_unchanged() -> None:
    session = ChatSession(FakeRetriever([]), settings=_settings(), client=FakeClient())
    assert session._max_deep_tokens() == 600_000


def test_agentaus_deep_refuses_over_derived_limit_mentions_context_window() -> None:
    settings = _agentaus_settings(agentaus_context_tokens=1_000, agentaus_max_output_tokens=100)
    limit = settings.agentaus_context_tokens - min(settings.chat_max_tokens, 100) - 4_000
    # Force a positive but small limit so a modest chunk exceeds it.
    assert limit <= 1_000  # sanity: floor kicks in, limit is the floor value
    huge_chunk = _deep_chunk(
        "huge", week=1, source_type=SourceType.PPTX, source_path="w1/a.pptx", text="x" * 20_000
    )
    chunk_source = FakeChunkSource([huge_chunk])
    client = FakeClient(stream_responses=[])
    session = ChatSession(
        FakeRetriever([]), settings=settings, client=client, chunk_source=chunk_source
    )

    with pytest.raises(ChatError, match="context window"):
        session.ask("question", mode="deep")

    assert client.messages.stream_calls == []


def test_deep_scope_switch_citations_map_to_new_scope_not_old() -> None:
    old_chunks = [
        _deep_chunk("old-a", week=1, source_type=SourceType.PPTX, source_path="w1/a.pptx"),
    ]
    new_chunks = [
        _deep_chunk("new-a", week=2, source_type=SourceType.PPTX, source_path="w2/a.pptx"),
    ]
    scoped_chunks = {
        (None, None, None): old_chunks,
    }

    def chunk_source(filters: SearchFilters | None) -> list[Chunk]:
        if filters is not None and filters.weeks == [2]:
            return new_chunks
        return scoped_chunks[(None, None, None)]

    turn1 = _citing_message(
        "About week 1.", search_result_index=0, cited_text="Some course material."
    )
    turn2 = _citing_message(
        "About week 2.", search_result_index=0, cited_text="Some course material."
    )
    client = FakeClient(stream_responses=[turn1, turn2])
    session = ChatSession(
        FakeRetriever([]), settings=_settings(), client=client, chunk_source=chunk_source
    )

    answer1 = session.ask("question about week 1", mode="deep")
    answer2 = session.ask("question about week 2", mode="deep", filters=SearchFilters(weeks=[2]))

    assert answer1.citations[0].chunk_id == "old-a"
    assert answer2.citations[0].chunk_id == "new-a"
    assert answer2.citations[0].source_path == "w2/a.pptx"
