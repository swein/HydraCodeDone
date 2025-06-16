import httpx
import logging
import json
from typing import List, Optional, Dict, Any, AsyncIterator

from ..models.openai_types import CustomChatCompletionMessage # For input
from openai.types.chat import ChatCompletionMessage as OpenAIChatCompletionMessage # For output

logger = logging.getLogger(__name__)

class LLMConnectionError(Exception):
    """Custom exception for LLM connection issues."""
    pass

class LLMResponseError(Exception):
    """Custom exception for unexpected LLM response format."""
    pass

async def call_ollama_chat_model(
    messages: List[CustomChatCompletionMessage], 
    model_name: str, 
    ollama_base_url: str,
    temperature: Optional[float] = 0.7,
    # stream: bool = False, # Ollama stream support might differ from OpenAI's
    # options: Optional[Dict[str, Any]] = None 
) -> OpenAIChatCompletionMessage: # Return the official OpenAI response message type
    logger.debug(f"Calling Ollama model '{model_name}' at {ollama_base_url} with {len(messages)} messages.")
    """
    Calls the Ollama /api/chat endpoint with the given messages and model.

    Args:
        messages: A list of CustomChatCompletionMessage objects representing the conversation history.
        model_name: The name of the Ollama model to use (e.g., "llama3").
        ollama_base_url: The base URL of the Ollama server (e.g., "http://localhost:11434").
        temperature: The temperature for sampling, passed to Ollama's options.
        # options: Optional dictionary of additional Ollama options.

    Returns:
        A ChatMessage object containing the assistant's response.

    Raises:
        LLMConnectionError: If there's an issue connecting to the Ollama server.
        LLMResponseError: If the Ollama server returns an unexpected response format or an error.
    """
    api_url = f"{ollama_base_url.rstrip('/')}/api/chat"

    # Convert CustomChatCompletionMessage Pydantic objects to dictionaries for Ollama
    # Ollama expects a list of {'role': 'user', 'content': '...'}, etc.
    ollama_messages = []
    for msg in messages:
        msg_dict = {"role": msg.role, "content": msg.content}
        # Ollama might not support 'name', 'tool_calls', 'tool_call_id' directly in the same way
        # or might have its own way to handle them. For basic chat, role and content are key.
        # If 'name' is relevant for your Ollama setup, you might need to add it conditionally.
        ollama_messages.append(msg_dict)

    payload = {
        "model": model_name,
        "messages": ollama_messages,
        "stream": False, # For now, we handle non-streamed responses
    }
    if temperature is not None:
        payload.setdefault("options", {}).update({"temperature": temperature})
    
    # if options:
    #     payload.setdefault("options", {}).update(options)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=120.0) # Added a timeout
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses

    except httpx.RequestError as e:
        logger.error(f"httpx.RequestError calling Ollama model '{model_name}' at {api_url}: {e}", exc_info=True)
        # Handles connection errors, timeouts (excluding read timeouts if stream=True), etc.
        raise LLMConnectionError(f"Error connecting to Ollama at {api_url}: {e}") from e
    except httpx.HTTPStatusError as e:
        logger.error(f"httpx.HTTPStatusError calling Ollama model '{model_name}' at {api_url}. Status: {e.response.status_code}, Response: {e.response.text}", exc_info=True)
        # Handles HTTP error responses (4xx, 5xx)
        error_detail = e.response.text
        try:
            error_json = e.response.json()
            error_detail = error_json.get("error", error_detail)
        except ValueError:
            pass # Keep original text if not JSON
        raise LLMResponseError(f"Ollama API request failed with status {e.response.status_code} at {api_url}: {error_detail}") from e

    try:
        response_data = response.json()
        if response_data.get("error"):
            logger.error(f"Ollama API returned an error for model '{model_name}': {response_data['error']}. Payload: {payload}")
            raise LLMResponseError(f"Ollama API returned an error: {response_data['error']}")
        
        # Ollama's non-streaming response structure for /api/chat contains a 'message' object
        assistant_response_data = response_data.get("message")
        if not assistant_response_data or not isinstance(assistant_response_data, dict):
            logger.error(f"Unexpected response structure from Ollama model '{model_name}': 'message' field is missing or not a dict. Response: {response_data}")
            raise LLMResponseError(f"Unexpected response structure from Ollama: 'message' field is missing or not a dict. Response: {response_data}")

        role = assistant_response_data.get("role")
        content = assistant_response_data.get("content")

        if role != "assistant" or content is None:
            logger.error(f"Unexpected content in Ollama response message for model '{model_name}'. Expected role 'assistant' and non-null content. Got role '{role}', content: '{content}'. Response: {response_data}")
            raise LLMResponseError(
                f"Unexpected content in Ollama response message. Expected role 'assistant' and non-null content. "
                f"Got role '{role}', content: '{content}'. Response: {response_data}"
            )
        
        logger.debug(f"Successfully received response from Ollama model '{model_name}'. Role: {role}, Content length: {len(content) if content else 0}")
        # Construct the official OpenAI ChatCompletionMessage for the response
        return OpenAIChatCompletionMessage(role=role, content=content)

    except ValueError as e:  # JSONDecodeError is a subclass of ValueError
        logger.error(
            f"Failed to decode JSON response from Ollama model '{model_name}': {e}. Response text: {response.text}",
            exc_info=True,
        )
        raise LLMResponseError(
            f"Failed to decode JSON response from Ollama: {e}. Response text: {response.text}"
        ) from e
    except KeyError as e:
        logger.error(
            f"Missing expected key {e} in Ollama response for model '{model_name}': {response_data}",
            exc_info=True,
        )
        raise LLMResponseError(
            f"Missing expected key {e} in Ollama response: {response_data}"
        ) from e

# ---------------------------------------------------------------------------
# Streaming Support
# ---------------------------------------------------------------------------
async def stream_ollama_chat_model(
    messages: List[CustomChatCompletionMessage],
    model_name: str,
    ollama_base_url: str,
    temperature: Optional[float] = 0.7,
) -> AsyncIterator[dict]:
    """Stream responses from Ollama as they arrive.

    Yields each JSON chunk emitted by Ollama. Caller is responsible for
    converting these chunks to the desired wire format (e.g. OpenAI SSE).
    """
    api_url = f"{ollama_base_url.rstrip('/')}/api/chat"

    ollama_messages = [{"role": m.role, "content": m.content} for m in messages]

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": ollama_messages,
        "stream": True,
    }
    if temperature is not None:
        payload.setdefault("options", {}).update({"temperature": temperature})

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", api_url, json=payload, timeout=120.0) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue  # skip keep-alive blanks
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Failed to decode streaming line from Ollama: %s", line)
                        continue
                    yield chunk
    except httpx.RequestError as e:
        logger.error(f"httpx.RequestError (stream) calling Ollama model '{model_name}' at {api_url}: {e}")
        raise LLMConnectionError(f"Error connecting to Ollama at {api_url}: {e}") from e
    except httpx.HTTPStatusError as e:
        error_detail = e.response.text
        try:
            error_json = e.response.json()
            error_detail = error_json.get("error", error_detail)
        except ValueError:
            pass
        logger.error(
            "httpx.HTTPStatusError (stream) calling Ollama model '%s'. Status: %s, Response: %s",
            model_name,
            e.response.status_code,
            error_detail,
        )
        raise LLMResponseError(
            f"Ollama API request failed with status {e.response.status_code} at {api_url}: {error_detail}"
        )


