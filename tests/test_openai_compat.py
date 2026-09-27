"""Tests for notecast.providers.openai_compat.OpenAICompatClient: the
Anthropic-shaped adapter over an OpenAI-compatible chat.completions API,
with a fully fake `openai` client — no network access.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest
from pydantic import BaseModel

from notecast.chat.client import ChatError, get_client
from notecast.chat.grounding import hits_to_search_results, parse_response
from notecast.chat.session import ChatSession
from notecast.config import Settings
from notecast.models import Chunk, Location, SearchHit, SourceType
from notecast.providers.openai_compat import (
    OpenAICompatClient,
    _strip_leaked_reasoning,
    map_openai_error,
)

# --- Fakes ---------------------------------------------------------------


def _completion(
    text: str | None,
    *,
    finish_reason: str = "stop",
    usage: dict[str, Any] | None = None,
    model: str = "agentaus-model",
    completion_id: str = "cmpl-1",
) -> SimpleNamespace:
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage_obj = None
    if usage is not None:
        details = usage.pop("prompt_tokens_details", None)
        usage_obj = SimpleNamespace(**usage)
        if details is not None:
            usage_obj.prompt_tokens_details = SimpleNamespace(**details)
    return SimpleNamespace(choices=[choice], usage=usage_obj, id=completion_id, model=model)


class FakeCompletions:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeChat:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeModelsApi:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def list(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(id=i) for i in self._ids]


class FakeOpenAI:
    def __init__(
        self, responses: list[Any] | None = None, model_ids: list[str] | None = None
    ) -> None:
        self.completions = FakeCompletions(responses)
        self.chat = FakeChat(self.completions)
        self.models = FakeModelsApi(model_ids or [])


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = dict(
        _env_file=None,
        provider="agentaus",
        agentaus_api_key="test-key",
        agentaus_base_url="https://agentaus.example.com/v1",
        agentaus_model="agentaus-model",
        agentaus_helper_model="agentaus-helper",
        agentaus_max_output_tokens=8192,
        agentaus_json_mode=True,
        agentaus_timeout_seconds=60.0,
        notebooks_dir="notebooks",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _response(status_code: int, headers: dict | None = None) -> httpx2.Response:
    request = httpx2.Request("POST", "https://agentaus.example.com/v1/chat/completions")
    return httpx2.Response(status_code, request=request, headers=headers or {})


def _hit(chunk_id: str, text: str, title: str = "Week 3 slides") -> SearchHit:
    chunk = Chunk(
        chunk_id=chunk_id,
        course="comp4650",
        source_path="week-03/lecture-2.pptx",
        source_type=SourceType.PPTX,
        ordinal=0,
        text=text,
        header=title,
        location=Location(slide=4),
        week=3,
    )
    return SearchHit(chunk=chunk, score=0.9)


# --- System / message flattening -----------------------------------------


def test_system_string_passthrough() -> None:
    fake = FakeOpenAI(responses=[_completion("hi there")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    client.messages.create(
        model="agentaus-model", max_tokens=100, system="Be helpful.", messages=[]
    )
    request = fake.completions.calls[0]
    assert request["messages"][0] == {"role": "system", "content": "Be helpful."}


def test_system_block_list_flattened_and_cache_control_ignored() -> None:
    fake = FakeOpenAI(responses=[_completion("hi")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    system = [
        {"type": "text", "text": "Rule one.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Rule two."},
    ]
    client.messages.create(model="agentaus-model", max_tokens=100, system=system, messages=[])
    request = fake.completions.calls[0]
    assert request["messages"][0]["content"] == "Rule one.\n\nRule two."


def test_plain_text_message_passthrough_no_sources() -> None:
    fake = FakeOpenAI(responses=[_completion("An answer with no citations.")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": "Hello?"}],
    )
    request = fake.completions.calls[0]
    assert request["messages"][-1] == {"role": "user", "content": "Hello?"}
    assert len(message.content) == 1
    assert message.content[0].text == "An answer with no citations."
    assert message.content[0].citations is None


# --- system_prompt_overwrite -----------------------------------------------


def test_system_prompt_overwrite_sent_when_enabled() -> None:
    fake = FakeOpenAI(responses=[_completion("hi")])
    settings = _settings(agentaus_system_prompt_overwrite=True)
    client = OpenAICompatClient(settings, openai_client=fake)
    client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="Be helpful.",
        messages=[{"role": "user", "content": "Hello?"}],
    )
    request = fake.completions.calls[0]
    assert request["extra_body"] == {"system_prompt_overwrite": True}


def test_system_prompt_overwrite_not_sent_when_disabled() -> None:
    fake = FakeOpenAI(responses=[_completion("hi")])
    client = OpenAICompatClient(
        _settings(agentaus_system_prompt_overwrite=False), openai_client=fake
    )
    client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="Be helpful.",
        messages=[{"role": "user", "content": "Hello?"}],
    )
    request = fake.completions.calls[0]
    assert "extra_body" not in request


def test_system_prompt_overwrite_adds_minimal_system_prompt_when_none_given() -> None:
    fake = FakeOpenAI(responses=[_completion("hi")])
    settings = _settings(agentaus_system_prompt_overwrite=True)
    client = OpenAICompatClient(settings, openai_client=fake)
    client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello?"}],
    )
    request = fake.completions.calls[0]
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][0]["content"]
    assert request["extra_body"] == {"system_prompt_overwrite": True}


def test_no_system_prompt_added_when_overwrite_disabled_and_none_given() -> None:
    fake = FakeOpenAI(responses=[_completion("hi")])
    client = OpenAICompatClient(
        _settings(agentaus_system_prompt_overwrite=False), openai_client=fake
    )
    client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello?"}],
    )
    request = fake.completions.calls[0]
    assert request["messages"][0]["role"] == "user"


# --- search_result numbering across multiple messages (deep mode shape) --


def test_search_result_numbering_runs_across_messages() -> None:
    fake = FakeOpenAI(responses=[_completion("ok")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    hit1 = _hit("c1", "First chunk text.", title="Source A")
    hit2 = _hit("c2", "Second chunk text.", title="Source B")

    # Deep-mode shape: material attached to an earlier user turn, later
    # turns are plain text, final turn carries the new question.
    messages = [
        {
            "role": "user",
            "content": [*hits_to_search_results([hit1]), {"type": "text", "text": "Q1"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "A1"}]},
        {
            "role": "user",
            "content": [*hits_to_search_results([hit2]), {"type": "text", "text": "Q2"}],
        },
    ]
    client.messages.create(model="agentaus-model", max_tokens=100, system="sys", messages=messages)
    request = fake.completions.calls[0]

    first_user = request["messages"][1]["content"]
    third_user = request["messages"][3]["content"]
    assert 'id="1"' in first_user
    assert 'id="2"' in third_user
    assert "First chunk text." in first_user
    assert "Second chunk text." in third_user
    # A citation instruction was appended since search_result blocks exist.
    assert "square brackets" in request["messages"][0]["content"]


# --- Citation splitting ----------------------------------------------------


def test_citation_splitting_single_and_double_markers() -> None:
    hit1 = _hit("c1", "Content about gradient descent.")
    hit2 = _hit("c2", "Content about learning rate.")
    sources = hits_to_search_results([hit1, hit2])
    fake = FakeOpenAI(
        responses=[_completion("Gradient descent minimises loss [1]. It uses a learning rate [2].")]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    blocks = message.content
    assert blocks[0].text == "Gradient descent minimises loss "
    assert blocks[0].citations[0].search_result_index == 0
    assert blocks[1].text == ". It uses a learning rate "
    assert blocks[1].citations[0].search_result_index == 1
    assert blocks[2].text == "."


def test_citation_splitting_adjacent_double_marker() -> None:
    hit1 = _hit("c1", "Content one.")
    hit2 = _hit("c2", "Content two.")
    sources = hits_to_search_results([hit1, hit2])
    fake = FakeOpenAI(responses=[_completion("Both apply here [2][5-of-range-ignored].")])
    # Use a real double marker instead in a second, cleaner test below.
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    # "[5-of-range-ignored]" isn't a valid numeric marker shape and is left alone by our
    # regex (it requires only digits/commas/hyphens between digits), so only [2] is
    # recognised here; this asserts we didn't crash on the odd bracket text.
    assert any(b.citations for b in message.content)


def test_citation_splitting_double_marker_and_comma_list() -> None:
    hits = [_hit(f"c{i}", f"Content {i}.") for i in range(1, 6)]
    sources = hits_to_search_results(hits)
    fake = FakeOpenAI(
        responses=[_completion("First point [2][5]. Second point [2, 5]. Done [2,5].")]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    blocks = message.content
    assert blocks[0].text == "First point "
    assert {c.search_result_index for c in blocks[0].citations} == {1, 4}
    assert blocks[1].text == ". Second point "
    assert {c.search_result_index for c in blocks[1].citations} == {1, 4}
    assert blocks[2].text == ". Done "
    assert {c.search_result_index for c in blocks[2].citations} == {1, 4}
    assert blocks[3].text == "."


def test_citation_splitting_out_of_range_ids_stripped_not_cited() -> None:
    hit1 = _hit("c1", "Only source.")
    sources = hits_to_search_results([hit1])
    fake = FakeOpenAI(responses=[_completion("A claim [1]. A bad claim [99].")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    full_text = "".join(b.text for b in message.content)
    assert "[99]" not in full_text
    assert "[1]" not in full_text
    assert any(b.citations for b in message.content)
    non_cited = [b for b in message.content if not b.citations]
    assert any(". A bad claim " in b.text for b in non_cited)


def test_not_in_sources_token_kept_at_start_of_first_block() -> None:
    hit1 = _hit("c1", "Some content.")
    sources = hits_to_search_results([hit1])
    fake = FakeOpenAI(
        responses=[
            _completion("[NOT_IN_SOURCES] The material doesn't cover this. See [1] instead.")
        ]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    assert message.content[0].text.startswith("[NOT_IN_SOURCES]")


def test_end_to_end_not_in_sources_via_grounding_parse_response() -> None:
    hit1 = _hit("c1", "Some content.")
    sources = hits_to_search_results([hit1])
    fake = FakeOpenAI(responses=[_completion("[NOT_IN_SOURCES] The material doesn't cover this.")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    segments, citations, not_in_sources = parse_response(message, [hit1])
    assert not_in_sources is True
    assert segments[0].text == "The material doesn't cover this."


# --- Citation markers vs. code indexing / code spans -----------------------


def test_code_indexing_not_mistaken_for_citations() -> None:
    hit1 = _hit("c1", "First source content.")
    hit2 = _hit("c2", "Second source content.")
    sources = hits_to_search_results([hit1, hit2])
    fake = FakeOpenAI(
        responses=[
            _completion(
                "Then x[2]=arr[1]+1. Chained access looks like matrix[1][2] and foo()[1].bar too."
            )
        ]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    # Every bracket here is followed by a code continuation (=, +, [, .bar),
    # so none of them were treated as citations: one uncited block, brackets
    # intact.
    assert len(message.content) == 1
    assert message.content[0].citations is None
    assert message.content[0].text == (
        "Then x[2]=arr[1]+1. Chained access looks like matrix[1][2] and foo()[1].bar too."
    )


def test_prose_touching_brackets_is_still_a_citation_even_after_an_identifier() -> None:
    """False negatives on prose are worse than rare false positives (code
    is separately protected by fenced/inline spans) — so a bracket that
    merely touches an identifier or closing paren, with nothing code-like
    *after* it, is still read as a citation.
    """
    hit1 = _hit("c1", "First source content.")
    hit2 = _hit("c2", "Second source content.")
    sources = hits_to_search_results([hit1, hit2])
    fake = FakeOpenAI(
        responses=[
            _completion("The complexity is O(n)[1]. This déjà vu[2] confirms the set[1] result.")
        ]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    full_text = "".join(b.text for b in message.content)
    assert "[1]" not in full_text
    assert "[2]" not in full_text
    cited_indices = {
        c.search_result_index for b in message.content if b.citations for c in b.citations
    }
    assert cited_indices == {0, 1}


def test_real_citations_still_work_alongside_prose_that_touches_brackets() -> None:
    hit1 = _hit("c1", "First source content.")
    hit2 = _hit("c2", "Second source content.")
    sources = hits_to_search_results([hit1, hit2])
    fake = FakeOpenAI(
        responses=[_completion("Gradient descent minimises loss.[2] Loss [1] is the objective.")]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    blocks = message.content
    assert blocks[0].text == "Gradient descent minimises loss."
    assert blocks[0].citations[0].search_result_index == 1
    assert blocks[1].text == " Loss "
    assert blocks[1].citations[0].search_result_index == 0
    assert blocks[2].text == " is the objective."


# --- Leading markers (regression: "" was matching `in ")]"`) ---------------


def test_leading_single_marker_at_start_of_text_is_a_citation() -> None:
    hit1 = _hit("c1", "Some content.")
    sources = hits_to_search_results([hit1])
    fake = FakeOpenAI(responses=[_completion("[1] According to the slides, X happens.")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    full_text = "".join(b.text for b in message.content)
    assert "[1]" not in full_text
    assert message.content[0].citations is not None
    assert message.content[0].citations[0].search_result_index == 0


def test_leading_double_marker_at_start_of_text_is_a_citation() -> None:
    hit1 = _hit("c1", "First.")
    hit2 = _hit("c2", "Second.")
    sources = hits_to_search_results([hit1, hit2])
    fake = FakeOpenAI(responses=[_completion("[1][2] Both sources apply here.")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    assert message.content[0].citations is not None
    assert {c.search_result_index for c in message.content[0].citations} == {0, 1}


def test_leading_comma_list_marker_at_start_of_text_is_a_citation() -> None:
    hits = [_hit(f"c{i}", f"Content {i}.") for i in range(1, 6)]
    sources = hits_to_search_results(hits)
    fake = FakeOpenAI(responses=[_completion("[2, 5] are the relevant sources.")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    assert message.content[0].citations is not None
    assert {c.search_result_index for c in message.content[0].citations} == {1, 4}


def test_markers_inside_inline_code_span_left_untouched() -> None:
    hit1 = _hit("c1", "Some content.")
    sources = hits_to_search_results([hit1])
    fake = FakeOpenAI(
        responses=[_completion("See `weights[1]` in the code, but the concept is cited [1].")]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    full_text = "".join(b.text for b in message.content)
    assert "`weights[1]`" in full_text
    cited_blocks = [b for b in message.content if b.citations]
    assert len(cited_blocks) == 1
    assert cited_blocks[0].citations[0].search_result_index == 0


def test_markers_inside_fenced_code_block_left_untouched() -> None:
    hit1 = _hit("c1", "Some content.")
    sources = hits_to_search_results([hit1])
    code_block = "```python\narr[1] = arr[2]\n```"
    fake = FakeOpenAI(
        responses=[_completion(f"Here is the snippet:\n{code_block}\nSee source [1].")]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    full_text = "".join(b.text for b in message.content)
    assert code_block in full_text
    cited_blocks = [b for b in message.content if b.citations]
    assert len(cited_blocks) == 1
    assert cited_blocks[0].citations[0].search_result_index == 0


def test_citation_between_two_fenced_code_blocks_still_works() -> None:
    """A stray backtick from one fenced block's ``` fence must not make the
    inline-code regex swallow the real citation (or the next fenced block)
    between them.
    """
    hit1 = _hit("c1", "First.")
    hit2 = _hit("c2", "Second.")
    sources = hits_to_search_results([hit1, hit2])
    text = "```\ncode[1]\n``` and cite [2] then ```more[1]```"
    fake = FakeOpenAI(responses=[_completion(text)])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    full_text = "".join(b.text for b in message.content)
    # The real "[2]" citation marker is stripped; the code-block "[1]"s
    # (inside fences) are left exactly as they were.
    assert full_text == text.replace("[2]", "")
    assert "code[1]" in full_text
    assert "more[1]" in full_text
    cited_blocks = [b for b in message.content if b.citations]
    assert len(cited_blocks) == 1
    assert cited_blocks[0].citations[0].search_result_index == 1


# --- <think> stripping ------------------------------------------------------


def test_think_block_stripped() -> None:
    fake = FakeOpenAI(responses=[_completion("<think>internal musing</think>The answer.")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == "The answer."


def test_unterminated_think_block_left_untouched() -> None:
    fake = FakeOpenAI(responses=[_completion("<think>never closes the answer")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == "<think>never closes the answer"


# --- Raw control-token and plain-text reasoning leak stripping -------------


def test_control_token_leak_cuts_reply_at_first_marker() -> None:
    fake = FakeOpenAI(
        responses=[_completion("The answer is 42.<|start|>assistant<|channel|>commentary blah")]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == "The answer is 42."


def test_control_token_leak_at_start_leaves_empty_text() -> None:
    fake = FakeOpenAI(responses=[_completion("<|start|>assistant<|channel|>commentary blah")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == ""


def test_leaked_reasoning_paragraph_stripped() -> None:
    fake = FakeOpenAI(
        responses=[_completion("We should use web search.\nActually, the answer is 42.")]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == "Actually, the answer is 42."


@pytest.mark.parametrize(
    "leading_line",
    [
        "I need to check the sources first.",
        "The user asks about gradient descent.",
        "Let's think about how to answer the question.",
        "Okay, we should answer directly.",
        "So, the user wants a definition.",
    ],
)
def test_various_leaked_reasoning_openers_stripped(leading_line: str) -> None:
    fake = FakeOpenAI(responses=[_completion(f"{leading_line}\nThe real answer follows.")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == "The real answer follows."


def test_legitimate_answer_starting_with_we_define_not_stripped() -> None:
    fake = FakeOpenAI(
        responses=[_completion("We define entropy as a measure of uncertainty.\nMore detail.")]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == (
        "We define entropy as a measure of uncertainty.\nMore detail."
    )


def test_legitimate_answer_starting_with_i_should_note_not_stripped() -> None:
    """ "I should note that ..." matches the modal opener but doesn't
    reference the task/tooling, so it's a real answer, not a leak.
    """
    fake = FakeOpenAI(
        responses=[_completion("I should note that the lecture defines X.\nMore text follows.")]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == (
        "I should note that the lecture defines X.\nMore text follows."
    )


def test_legitimate_answer_starting_with_we_will_see_not_stripped() -> None:
    fake = FakeOpenAI(
        responses=[_completion("We will see that entropy measures uncertainty.\nMore detail.")]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == (
        "We will see that entropy measures uncertainty.\nMore detail."
    )


def test_leaked_opener_with_task_keyword_web_search_still_stripped() -> None:
    fake = FakeOpenAI(responses=[_completion("We should use web search.\nAnswer follows.")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == "Answer follows."


def test_leaked_opener_with_task_keywords_answer_and_sources_still_stripped() -> None:
    fake = FakeOpenAI(
        responses=[
            _completion(
                "I need to answer the question using the sources.\nHere is the real answer."
            )
        ]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == "Here is the real answer."


def test_leading_line_with_citation_marker_not_stripped() -> None:
    hit1 = _hit("c1", "Some content.")
    sources = hits_to_search_results([hit1])
    fake = FakeOpenAI(
        responses=[_completion("We can see from slide 3 [1] that this holds.\nMore detail.")]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": [*sources, {"type": "text", "text": "Q"}]}],
    )
    full_text = "".join(b.text for b in message.content)
    assert full_text.startswith("We can see from slide 3 ")


def test_leaked_reasoning_not_stripped_when_nothing_follows() -> None:
    fake = FakeOpenAI(responses=[_completion("We should use web search.")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == "We should use web search."


def test_leaked_reasoning_not_stripped_when_only_blank_line_follows() -> None:
    fake = FakeOpenAI(responses=[_completion("We should use web search.\n   \n")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == "We should use web search.\n   \n"


def test_leaked_reasoning_not_stripped_when_no_trailing_period() -> None:
    fake = FakeOpenAI(responses=[_completion("We should use web search\nActual answer follows.")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == "We should use web search\nActual answer follows."


def test_leaked_reasoning_not_stripped_when_over_max_len() -> None:
    long_line = "We should " + ("really " * 60) + "search for this."
    assert len(long_line) > 300
    fake = FakeOpenAI(responses=[_completion(f"{long_line}\nActual answer follows.")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.content[0].text == f"{long_line}\nActual answer follows."


# --- max_tokens clamping ----------------------------------------------------


def test_max_tokens_clamped_to_settings_cap() -> None:
    fake = FakeOpenAI(responses=[_completion("ok")])
    client = OpenAICompatClient(_settings(agentaus_max_output_tokens=500), openai_client=fake)
    client.messages.create(model="agentaus-model", max_tokens=100_000, system="sys", messages=[])
    assert fake.completions.calls[0]["max_tokens"] == 500


def test_max_tokens_under_cap_passed_through() -> None:
    fake = FakeOpenAI(responses=[_completion("ok")])
    client = OpenAICompatClient(_settings(agentaus_max_output_tokens=5000), openai_client=fake)
    client.messages.create(model="agentaus-model", max_tokens=100, system="sys", messages=[])
    assert fake.completions.calls[0]["max_tokens"] == 100


# --- Tools / web_search dropped ---------------------------------------------


def test_web_search_tool_dropped_and_note_added() -> None:
    fake = FakeOpenAI(responses=[_completion("ok")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]
    client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", tools=tools, messages=[]
    )
    request = fake.completions.calls[0]
    assert "tools" not in request
    assert "unavailable" in request["messages"][0]["content"]


# --- output_config json_schema -----------------------------------------------


def test_json_schema_output_config_adds_schema_and_response_format() -> None:
    fake = FakeOpenAI(responses=[_completion('{"answer": "42"}')])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[],
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": schema}},
    )
    request = fake.completions.calls[0]
    assert "JSON Schema" in request["messages"][0]["content"]
    assert request["response_format"] == {"type": "json_object"}
    assert message.content[0].text == '{"answer": "42"}'
    assert message.content[0].citations is None


def test_json_mode_disabled_never_sends_response_format() -> None:
    fake = FakeOpenAI(responses=[_completion('{"answer": "42"}')])
    client = OpenAICompatClient(_settings(agentaus_json_mode=False), openai_client=fake)
    schema = {"type": "object"}
    client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    request = fake.completions.calls[0]
    assert "response_format" not in request


def test_response_format_400_falls_back_and_is_remembered() -> None:
    bad_request = openai.BadRequestError(
        "response_format not supported", response=_response(400), body=None
    )
    fake = FakeOpenAI(
        responses=[
            bad_request,
            _completion('{"ok": true}'),
            _completion('{"ok": true}'),
        ]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    schema = {"type": "object"}
    kwargs = dict(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )

    message = client.messages.create(**kwargs)
    assert message.content[0].text == '{"ok": true}'
    assert "response_format" not in fake.completions.calls[1]

    # Second call: never sends response_format again, no retry needed.
    client.messages.create(**kwargs)
    assert len(fake.completions.calls) == 3
    assert "response_format" not in fake.completions.calls[2]


def test_json_retry_appends_assistant_reply_and_fix_instruction() -> None:
    fake = FakeOpenAI(
        responses=[_completion("here is some prose, not JSON"), _completion('{"ok": true}')]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    schema = {"type": "object"}
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": "Give me JSON."}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    assert message.content[0].text == '{"ok": true}'
    assert len(fake.completions.calls) == 2
    retry_messages = fake.completions.calls[1]["messages"]
    assert retry_messages[-2] == {
        "role": "assistant",
        "content": "here is some prose, not JSON",
    }
    assert retry_messages[-1]["role"] == "user"
    assert "not valid JSON" in retry_messages[-1]["content"]


def test_json_retry_sums_usage_across_both_calls() -> None:
    fake = FakeOpenAI(
        responses=[
            _completion(
                "prose",
                usage={"input_tokens": 100, "output_tokens": 10},
            ),
            _completion(
                '{"ok": true}',
                usage={"input_tokens": 150, "output_tokens": 20},
            ),
        ]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    schema = {"type": "object"}
    message = client.messages.create(
        model="agentaus-model",
        max_tokens=100,
        system="sys",
        messages=[],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    assert message.usage.input_tokens == 250
    assert message.usage.output_tokens == 30


def test_json_retry_with_output_format_succeeds_on_second_attempt() -> None:
    fake = FakeOpenAI(responses=[_completion("nope"), _completion('{"answer": "42"}')])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.parse(
        model="agentaus-model", max_tokens=100, system="sys", messages=[], output_format=_Answer
    )
    assert message.parsed_output == _Answer(answer="42")
    assert len(fake.completions.calls) == 2


def test_no_json_retry_when_first_attempt_already_valid() -> None:
    fake = FakeOpenAI(responses=[_completion('{"answer": "42"}')])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.parse(
        model="agentaus-model", max_tokens=100, system="sys", messages=[], output_format=_Answer
    )
    assert message.parsed_output == _Answer(answer="42")
    assert len(fake.completions.calls) == 1


# --- parse() / stream(output_format=...) ------------------------------------


class _Answer(BaseModel):
    answer: str


def test_parse_returns_parsed_output_on_valid_json() -> None:
    fake = FakeOpenAI(responses=[_completion('{"answer": "42"}')])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.parse(
        model="agentaus-model", max_tokens=100, system="sys", messages=[], output_format=_Answer
    )
    assert message.parsed_output == _Answer(answer="42")


def test_parse_handles_code_fenced_json() -> None:
    fake = FakeOpenAI(responses=[_completion('```json\n{"answer": "42"}\n```')])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.parse(
        model="agentaus-model", max_tokens=100, system="sys", messages=[], output_format=_Answer
    )
    assert message.parsed_output == _Answer(answer="42")


def test_parse_returns_none_on_invalid_json_after_retry() -> None:
    fake = FakeOpenAI(responses=[_completion("not json at all"), _completion("still not json")])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.parse(
        model="agentaus-model", max_tokens=100, system="sys", messages=[], output_format=_Answer
    )
    assert message.parsed_output is None
    assert len(fake.completions.calls) == 2


def test_stream_output_format_via_context_manager() -> None:
    fake = FakeOpenAI(responses=[_completion('{"answer": "streamed"}')])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    with client.messages.stream(
        model="agentaus-model", max_tokens=100, system="sys", messages=[], output_format=_Answer
    ) as stream:
        message = stream.get_final_message()
    assert message.parsed_output == _Answer(answer="streamed")


# --- stop_reason mapping ------------------------------------------------------


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("content_filter", "refusal"),
        ("tool_calls", "tool_use"),
        (None, "end_turn"),
        ("something_else", "end_turn"),
    ],
)
def test_stop_reason_mapping(finish_reason: str | None, expected: str) -> None:
    fake = FakeOpenAI(responses=[_completion("ok", finish_reason=finish_reason)])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.stop_reason == expected
    assert message.stop_details is None


# --- usage mapping -------------------------------------------------------------


def test_usage_mapping_with_cache_read_tokens() -> None:
    fake = FakeOpenAI(
        responses=[
            _completion(
                "ok",
                usage={
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "prompt_tokens_details": {"cached_tokens": 40},
                },
            )
        ]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.usage.input_tokens == 120
    assert message.usage.output_tokens == 30
    assert message.usage.cache_read_input_tokens == 40
    assert message.usage.cache_creation_input_tokens == 0
    assert message.usage.server_tool_use is None


def test_usage_missing_defaults_to_zeros() -> None:
    fake = FakeOpenAI(responses=[_completion("ok", usage=None)])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.usage.input_tokens == 0
    assert message.usage.output_tokens == 0
    assert message.usage.cache_read_input_tokens == 0


def test_usage_prefers_agentaus_input_output_tokens_over_openai_names() -> None:
    fake = FakeOpenAI(
        responses=[
            _completion(
                "ok",
                usage={
                    "input_tokens": 120,
                    "output_tokens": 45,
                    "prompt_tokens": 999,
                    "completion_tokens": 999,
                },
            )
        ]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.usage.input_tokens == 120
    assert message.usage.output_tokens == 45


def test_usage_falls_back_to_openai_names_when_agentaus_names_absent() -> None:
    fake = FakeOpenAI(
        responses=[_completion("ok", usage={"prompt_tokens": 200, "completion_tokens": 60})]
    )
    client = OpenAICompatClient(_settings(), openai_client=fake)
    message = client.messages.create(
        model="agentaus-model", max_tokens=100, system="sys", messages=[]
    )
    assert message.usage.input_tokens == 200
    assert message.usage.output_tokens == 60


def test_usage_input_output_tokens_read_via_real_openai_sdk_round_trip() -> None:
    """Proves this works end-to-end against the real `openai` SDK — a real
    `openai.OpenAI` client, over a mocked HTTP transport, whose server
    response carries ONLY `input_tokens`/`output_tokens` (no
    `prompt_tokens`/`completion_tokens`/`total_tokens` at all, matching what
    AgentAUS actually returns) — not just our fakes or a directly
    `model_validate`-d payload. Also proves the request itself: the URL and
    that `system_prompt_overwrite` rides as a top-level body field (how
    `extra_body` actually serialises through the real SDK), not nested.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        payload = {
            "id": "cmpl-real-1",
            "object": "chat.completion",
            "created": 0,
            "model": "agentaus.v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "The answer is 42."},
                }
            ],
            "usage": {"input_tokens": 120, "output_tokens": 45},
        }
        return httpx2.Response(200, json=payload, request=request)

    real_openai_client = openai.OpenAI(
        api_key="test-key",
        base_url="https://agentaus.com.au/api/v1",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    settings = _settings(
        agentaus_base_url="https://agentaus.com.au/api/v1", agentaus_system_prompt_overwrite=True
    )
    client = OpenAICompatClient(settings, openai_client=real_openai_client)

    message = client.messages.create(model="agentaus.v1", max_tokens=100, system="sys", messages=[])

    assert message.usage.input_tokens == 120
    assert message.usage.output_tokens == 45
    assert message.content[0].text == "The answer is 42."
    assert captured["url"] == "https://agentaus.com.au/api/v1/chat/completions"
    assert captured["body"]["system_prompt_overwrite"] is True


# --- list_models ---------------------------------------------------------------


def test_list_models() -> None:
    fake = FakeOpenAI(model_ids=["model-a", "model-b"])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    assert client.list_models() == ["model-a", "model-b"]


# --- Error mapping --------------------------------------------------------------


def test_map_authentication_error() -> None:
    exc = openai.AuthenticationError("bad key", response=_response(401), body=None)
    mapped = map_openai_error(exc)
    assert isinstance(mapped, ChatError)
    assert "AGENTAUS_API_KEY" in str(mapped)


def test_map_not_found_error() -> None:
    exc = openai.NotFoundError("model not found", response=_response(404), body=None)
    mapped = map_openai_error(exc)
    assert "AGENTAUS_MODEL" in str(mapped) or "AGENTAUS_BASE_URL" in str(mapped)


def test_map_rate_limit_error_includes_retry_after() -> None:
    exc = openai.RateLimitError(
        "too many requests", response=_response(429, {"retry-after": "7"}), body=None
    )
    mapped = map_openai_error(exc)
    assert "7" in str(mapped)


def test_map_server_error() -> None:
    exc = openai.InternalServerError("boom", response=_response(500), body=None)
    mapped = map_openai_error(exc)
    assert "server error" in str(mapped).lower()


def test_map_connection_error() -> None:
    request = httpx2.Request("POST", "https://agentaus.example.com/v1/chat/completions")
    exc = openai.APIConnectionError(request=request)
    mapped = map_openai_error(exc)
    assert "connect" in str(mapped).lower()


def test_map_timeout_error() -> None:
    request = httpx2.Request("POST", "https://agentaus.example.com/v1/chat/completions")
    exc = openai.APITimeoutError(request=request)
    mapped = map_openai_error(exc)
    assert "connect" in str(mapped).lower()


def test_map_unknown_error_still_returns_chat_error() -> None:
    mapped = map_openai_error(ValueError("something else"))
    assert isinstance(mapped, ChatError)


def test_error_raised_by_create_is_mapped_to_chat_error() -> None:
    exc = openai.AuthenticationError("bad key", response=_response(401), body=None)
    fake = FakeOpenAI(responses=[exc])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    with pytest.raises(ChatError, match="AGENTAUS_API_KEY"):
        client.messages.create(model="agentaus-model", max_tokens=100, system="sys", messages=[])


def test_map_max_model_len_error_uses_configured_context_tokens() -> None:
    exc = openai.BadRequestError(
        "This model's maximum context length... max_model_len exceeded",
        response=_response(400),
        body=None,
    )
    settings = _settings(agentaus_context_tokens=131_072)
    mapped = map_openai_error(exc, settings)
    assert "131,072" in str(mapped)
    assert "context window" in str(mapped)
    assert "hidden web search" in str(mapped)


def test_map_max_model_len_error_without_settings_is_generic() -> None:
    exc = openai.BadRequestError("max_model_len exceeded", response=_response(400), body=None)
    mapped = map_openai_error(exc)
    assert "131k" in str(mapped)


def test_max_model_len_error_raised_by_create_is_mapped() -> None:
    exc = openai.BadRequestError(
        "max_model_len exceeded, please reduce your input", response=_response(400), body=None
    )
    fake = FakeOpenAI(responses=[exc])
    client = OpenAICompatClient(_settings(agentaus_context_tokens=131_072), openai_client=fake)
    with pytest.raises(ChatError, match="context window"):
        client.messages.create(model="agentaus-model", max_tokens=100, system="sys", messages=[])


# --- get_client dispatch ---------------------------------------------------------


def test_get_client_returns_openai_compat_client_for_agentaus_provider() -> None:
    from notecast.chat import client as client_module

    client_module._agentaus_clients.clear()
    settings = _settings()
    client = get_client(settings)
    assert isinstance(client, OpenAICompatClient)
    client_module._agentaus_clients.clear()


# --- Integration: ChatSession over the adapter -----------------------------


def _chat_session_settings() -> Settings:
    return _settings(
        chat_model="agentaus-model",
        helper_model="agentaus-helper",
        deep_model="agentaus-model",
        chat_effort="medium",
        chat_max_tokens=8000,
        web_search_max_uses=3,
        retrieval_top_k=5,
    )


class FakeRetriever:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits

    def search(self, query: str, k: int = 10, filters: Any = None, mode: str = "hybrid") -> Any:
        return self.hits


def test_chat_session_over_agentaus_adapter_with_citations() -> None:
    hit1 = _hit("c1", "Gradient descent content.", title="Week 3 slides")
    hit2 = _hit("c2", "Learning rate content.", title="Week 3 slides")
    fake = FakeOpenAI(
        responses=[
            _completion(
                "Gradient descent minimises loss [1]. It uses a learning rate [2].",
                usage={"prompt_tokens": 200, "completion_tokens": 40},
            )
        ]
    )
    settings = _chat_session_settings()
    client = OpenAICompatClient(settings, openai_client=fake)
    retriever = FakeRetriever([hit1, hit2])
    session = ChatSession(retriever, settings=settings, client=client)

    answer = session.ask("How does gradient descent work?", mode="sources")

    assert len(answer.citations) == 2
    course_citations = {c.chunk_id for c in answer.citations}
    assert course_citations == {"c1", "c2"}
    full_text = "".join(seg.text for seg in answer.segments)
    assert full_text == ("Gradient descent minimises loss . It uses a learning rate .")
    assert answer.not_in_sources is False


@pytest.mark.parametrize(
    "first_line",
    [
        "We can use binary search.",
        "I should point out the answer depends on the prior.",
        "We will see that the answer is 42.",
    ],
)
def test_content_sentences_with_generic_task_words_not_stripped(first_line: str) -> None:
    text = f"{first_line}\nMore content follows here."
    assert _strip_leaked_reasoning(text) == text


def test_non_chat_completion_response_raises_readable_chat_error() -> None:
    """A server reply the openai SDK can't parse into a ChatCompletion (e.g.
    a non-JSON or malformed body) surfaces as a readable ChatError instead
    of an AttributeError deep in response translation.
    """
    fake = FakeOpenAI(responses=["not a real completion body"])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    with pytest.raises(ChatError, match="AgentAUS's response wasn't in the expected format"):
        client.messages.create(
            model="agentaus.v1",
            max_tokens=20,
            system="hi",
            messages=[{"role": "user", "content": "ping"}],
        )


def test_non_chat_completion_response_includes_truncated_raw_text() -> None:
    fake = FakeOpenAI(responses=["x" * 800])
    client = OpenAICompatClient(_settings(), openai_client=fake)
    with pytest.raises(ChatError, match="truncated"):
        client.messages.create(
            model="agentaus.v1",
            max_tokens=20,
            system="hi",
            messages=[{"role": "user", "content": "ping"}],
        )
