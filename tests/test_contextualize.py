"""Tests for contextual retrieval (LLM context prepending)."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from emb.contextualize import (
    build_context_prompt,
    contextualize_entry,
    contextualize_batch,
    contextualize_sync,
)
from emb.schema import Entry


def _mock_litellm(response_text="This is about Python programming."):
    """Create a mock litellm module."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    mock.acompletion = AsyncMock(return_value=mock_response)
    return mock


def test_build_context_prompt():
    entry = Entry(id="x", text="Hello world", source="chatgpt", title="Test", date="2024-01-01")
    prompt = build_context_prompt(entry)
    assert "chatgpt" in prompt
    assert "Test" in prompt
    assert "Hello world" in prompt


def test_build_context_prompt_truncates_long_text():
    entry = Entry(id="x", text="a" * 5000, source="test")
    prompt = build_context_prompt(entry)
    # Text in prompt should be capped at 2000 chars
    assert len(prompt) < 3000


def test_build_context_prompt_handles_missing_fields():
    """Prompt should handle entries with no source/title/date."""
    entry = Entry(id="x", text="Just text")
    prompt = build_context_prompt(entry)
    assert "unknown" in prompt
    assert "untitled" in prompt
    assert "Just text" in prompt


def test_contextualize_entry():
    entry = Entry(id="x", text="Original text", source="test")
    mock_llm = _mock_litellm("This discusses testing.")

    result = asyncio.run(contextualize_entry(entry, "test-model", mock_llm))
    assert result.text.startswith("This discusses testing.")
    assert "Original text" in result.text
    assert result.metadata["contextualized"] is True
    assert result.metadata["context_model"] == "test-model"


def test_contextualize_entry_skips_already_done():
    entry = Entry(id="x", text="Already done", metadata={"contextualized": True})
    mock_llm = _mock_litellm()

    result = asyncio.run(contextualize_entry(entry, "test-model", mock_llm))
    assert result.text == "Already done"  # unchanged
    mock_llm.acompletion.assert_not_called()


def test_contextualize_entry_handles_error():
    entry = Entry(id="x", text="Original", source="test")
    mock_llm = MagicMock()
    mock_llm.acompletion = AsyncMock(side_effect=Exception("API error"))

    result = asyncio.run(contextualize_entry(entry, "test-model", mock_llm))
    assert result.text == "Original"  # unchanged on error


def test_contextualize_entry_preserves_existing_metadata():
    """Existing metadata fields should survive contextualization."""
    entry = Entry(id="x", text="Hello", source="test", metadata={"custom_key": 42})
    mock_llm = _mock_litellm("Context.")

    result = asyncio.run(contextualize_entry(entry, "test-model", mock_llm))
    assert result.metadata["custom_key"] == 42
    assert result.metadata["contextualized"] is True


def test_contextualize_entry_preserves_other_fields():
    """source, title, date, id should be preserved."""
    entry = Entry(id="myid", text="Hello", source="git", title="My Title", date="2025-01-01")
    mock_llm = _mock_litellm("Context.")

    result = asyncio.run(contextualize_entry(entry, "test-model", mock_llm))
    assert result.id == "myid"
    assert result.source == "git"
    assert result.title == "My Title"
    assert result.date == "2025-01-01"


def test_contextualize_batch():
    entries = [
        Entry(id="a", text="First entry", source="test"),
        Entry(id="b", text="Second entry", source="test"),
        Entry(id="c", text="Already done", metadata={"contextualized": True}),
    ]
    mock_llm = _mock_litellm("Context added.")

    results = asyncio.run(contextualize_batch(entries, "test-model", concurrency=2, litellm_module=mock_llm))
    assert len(results) == 3
    # a and b should be contextualized
    assert results[0].metadata.get("contextualized") is True
    assert results[1].metadata.get("contextualized") is True
    # c was already done
    assert results[2].text == "Already done"
    # LLM should have been called twice (not for c)
    assert mock_llm.acompletion.call_count == 2


def test_contextualize_batch_all_already_done():
    """When all entries are already contextualized, nothing should happen."""
    entries = [
        Entry(id="a", text="Done 1", metadata={"contextualized": True}),
        Entry(id="b", text="Done 2", metadata={"contextualized": True}),
    ]
    mock_llm = _mock_litellm()

    results = asyncio.run(contextualize_batch(entries, "test-model", litellm_module=mock_llm))
    assert len(results) == 2
    mock_llm.acompletion.assert_not_called()


def test_contextualize_sync():
    entries = [Entry(id="x", text="Hello", source="test")]
    mock_llm = _mock_litellm("Context.")

    results = contextualize_sync(entries, "test-model", litellm_module=mock_llm)
    assert len(results) == 1
    assert results[0].metadata["contextualized"] is True


def test_contextualize_preserves_order():
    entries = [Entry(id=f"e{i}", text=f"text {i}", source="test") for i in range(5)]
    mock_llm = _mock_litellm("Ctx.")

    results = asyncio.run(contextualize_batch(entries, "test-model", litellm_module=mock_llm))
    for i, r in enumerate(results):
        assert r.id == f"e{i}"


def test_contextualize_batch_empty_list():
    """Empty input should return empty output without error."""
    mock_llm = _mock_litellm()
    results = asyncio.run(contextualize_batch([], "test-model", litellm_module=mock_llm))
    assert results == []
    mock_llm.acompletion.assert_not_called()
