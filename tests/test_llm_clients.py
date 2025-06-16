"""Tests for llm_clients module (call_ollama_chat_model & stream_ollama_chat_model)."""

import json
import pytest
import pytest
respx = pytest.importorskip("respx")
import httpx
from openai.types.chat import ChatCompletionMessage as OpenAIChatCompletionMessage
from src.services.llm_clients import (
    call_ollama_chat_model,
    stream_ollama_chat_model,
)
from src.models.openai_types import CustomChatCompletionMessage

OLLAMA_URL = "http://ollama:11434"

@pytest.mark.asyncio
@respx.mock
async def test_call_ollama_chat_model_success():
    """call_ollama_chat_model should return assistant message when Ollama responds 200."""
    messages = [CustomChatCompletionMessage(role="user", content="Ping?")]
    expected_reply = "Pong!"
    respx.post(f"{OLLAMA_URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": expected_reply},
                "done": True,
            },
        )
    )

    result: OpenAIChatCompletionMessage = await call_ollama_chat_model(
        messages=messages,
        model_name="dummy",
        ollama_base_url=OLLAMA_URL,
    )

    assert result.role == "assistant"
    assert result.content == expected_reply


@pytest.mark.asyncio
@respx.mock
async def test_call_ollama_chat_model_http_error():
    """Should raise when Ollama returns non-200."""
    messages = [CustomChatCompletionMessage(role="user", content="hi")]
    respx.post(f"{OLLAMA_URL}/api/chat").mock(return_value=httpx.Response(500, json={"error": "boom"}))

    with pytest.raises(Exception):
        await call_ollama_chat_model(messages, "dummy", OLLAMA_URL)


@pytest.mark.asyncio
async def test_stream_ollama_chat_model_iterates(monkeypatch):
    """stream_ollama_chat_model should yield decoded JSON chunks."""
    messages = [CustomChatCompletionMessage(role="user", content="stream?")]

    # Build a fake AsyncClient.stream context manager
    class DummyStreamResponse:
        def __init__(self):
            self._lines = [
                json.dumps({"message": {"content": "Hello"}, "done": False}),
                json.dumps({"message": {"content": " world"}, "done": False}),
                json.dumps({"done": True}),
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def aiter_lines(self):
            for ln in self._lines:
                yield ln

        def raise_for_status(self):
            pass

    class DummyAsyncClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, *a, **kw):
            return DummyStreamResponse()

    monkeypatch.patch("src.services.llm_clients.httpx.AsyncClient", DummyAsyncClient)

    chunks = []
    async for ch in stream_ollama_chat_model(messages, "dummy", OLLAMA_URL):
        chunks.append(ch)
    assert chunks[-1]["done"] is True
    assert len(chunks) == 3
