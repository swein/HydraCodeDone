import logging
import uuid
import time
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest # Renamed to avoid conflict
from .models.openai_types import (
    ChatCompletionRequest, 
    OpenAIChatCompletion, 
    OpenAIChatCompletionChoice, 
    create_default_usage
)
from openai.types.chat import ChatCompletionMessage as OpenAIChatCompletionMessage # For type hinting internal messages
from .services.llm_clients import call_ollama_chat_model, LLMConnectionError, LLMResponseError
from .services.critique_service import prepare_critique_messages
from .core.config import settings
from .core.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HydraCodeDone LLM Critique Proxy",
    description="A proxy server to enhance AI-assisted coding with a dual-model critique pipeline.",
    version="0.1.0",
)

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}

@app.post("/v1/chat/completions", response_model=OpenAIChatCompletion, tags=["Chat"])
async def chat_completions(request: ChatCompletionRequest, http_request: FastAPIRequest): # Added http_request for client info
    client_host = http_request.client.host if http_request.client else "unknown_client"
    logger.info(f"Received chat completion request from {client_host} for model {request.model}")
    """
    OpenAI-compatible chat completions endpoint.
    This endpoint takes a user request, calls a primary LLM (Model 1 via Ollama),
    and returns its response. Critique (Model 2) is not yet implemented.
    """


    final_assistant_message: OpenAIChatCompletionMessage

    try:
        # Call Model 1 (Ollama)
        model_1_response_message = await call_ollama_chat_model(
            messages=request.messages,
            model_name=settings.PRIMARY_MODEL_NAME,
            ollama_base_url=settings.OLLAMA_BASE_URL,
            temperature=request.temperature
        )

    except LLMConnectionError as e:
        logger.error(f"LLMConnectionError for Model 1 from {client_host}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Error connecting to Primary LLM (Model 1): {e}")
    except LLMResponseError as e:
        logger.error(f"LLMResponseError for Model 1 from {client_host}: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Error receiving response from Primary LLM (Model 1): {e}")
    except Exception as e:
        logger.error(f"Unexpected error during Model 1 call from {client_host}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during Model 1 call: {e}")

    # Prepare messages for Model 2 (Critique Model)
    if not settings.CRITIQUE_MODEL_NAME or not settings.CRITIQUE_SYSTEM_PROMPT:
        # If critique model is not configured, just return Model 1's response
        final_assistant_message = model_1_response_message
    else:
        try:
            critique_messages = prepare_critique_messages(
                original_request_messages=request.messages, # Ensure this matches the param name in critique_service
                model1_response_message=model_1_response_message, # Ensure this matches the param name
                critique_task_instruction=settings.CRITIQUE_SYSTEM_PROMPT
            )
            
            # Call Model 2 (Critique Model)
            model_2_response_message = await call_ollama_chat_model(
                messages=critique_messages,
                model_name=settings.CRITIQUE_MODEL_NAME,
                ollama_base_url=settings.OLLAMA_BASE_URL, # Assuming critique model is on the same Ollama instance
                temperature=request.temperature # Or a different temperature for critique
            )
            final_assistant_message = model_2_response_message

        except LLMConnectionError as e:
            logger.error(f"LLMConnectionError for Model 2 from {client_host}: {e}. Model 1 response was: {model_1_response_message.content}", exc_info=True)
            # Consider if we should fallback to Model 1's response or error out
            # For now, error out, but log that Model 1 was successful.
            raise HTTPException(status_code=503, detail=f"Error connecting to Critique LLM (Model 2): {e}. Model 1 response was: {model_1_response_message.content}")
        except LLMResponseError as e:
            logger.error(f"LLMResponseError for Model 2 from {client_host}: {e}. Model 1 response was: {model_1_response_message.content}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"Error receiving response from Critique LLM (Model 2): {e}. Model 1 response was: {model_1_response_message.content}")
        except Exception as e:
            logger.error(f"Unexpected error during Model 2 call from {client_host}: {e}. Model 1 response was: {model_1_response_message.content}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during Model 2 call: {e}. Model 1 response was: {model_1_response_message.content}")

    # Create a choice containing the final assistant message (either from Model 1 or Model 2)
    choice = OpenAIChatCompletionChoice(
        index=0, 
        message=final_assistant_message, 
        finish_reason="stop", # Assuming 'stop' is appropriate; Ollama might provide other reasons
        logprobs=None  # Explicitly set to None if not available
    )
    
    # Construct the full OpenAI-compatible response object
    response = OpenAIChatCompletion(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model, # Echo back the requested model, or use the actual model name if it differs
        choices=[choice],
        usage=create_default_usage() # Use the helper for default token counts
        # system_fingerprint can be added if available/relevant
    )
    logger.info(f"Successfully processed chat completion request from {client_host} for model {request.model}. Returning refined response.")
    return response
