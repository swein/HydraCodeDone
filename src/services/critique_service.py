import logging # Add logging import
from typing import List

from ..models.openai_types import CustomChatCompletionMessage # For constructing new messages
from openai.types.chat import ChatCompletionMessage as OpenAIChatCompletionMessage # For model1_response_message type hint

logger = logging.getLogger(__name__) # Initialize logger

def prepare_critique_messages(
    original_request_messages: List[CustomChatCompletionMessage],
    model1_response_message: OpenAIChatCompletionMessage, # Model 1's response is an OpenAI official type
    critique_task_instruction: str, # This is the full instruction set for Model 2
) -> List[CustomChatCompletionMessage]: # Returns messages ready for the next LLM call
    """
    Prepares the list of messages to be sent to Model 2 (the critique model).
    This involves passing the full original conversation context, Model 1's response,
    and then the specific critique instructions as a final user message.

    Args:
        original_request_messages: The list of messages from the original user request.
        model1_response_message: The ChatCompletionMessage object containing the response from Model 1.
        critique_task_instruction: The detailed instructions for Model 2, telling it how to critique
                                     and refine Model 1's response while adhering to original request formats.

    Returns:
        A list of CustomChatCompletionMessage objects ready to be sent to Model 2.
    """
    critique_request_messages: List[CustomChatCompletionMessage] = []

    # 1. Add all original request messages
    # Ensure they are CustomChatCompletionMessage instances if they aren't already
    # (though they should be, based on type hints from main.py)
    for msg in original_request_messages:
        critique_request_messages.append(msg)

    # 2. Add Model 1's response as an assistant message
    # Convert OpenAIChatCompletionMessage to CustomChatCompletionMessage
    model1_content = model1_response_message.content if model1_response_message.content is not None else ""
    critique_request_messages.append(
        CustomChatCompletionMessage(role="assistant", content=model1_content)
    )

    # 3. Add the critique task instruction as a final user message
    critique_request_messages.append(
        CustomChatCompletionMessage(role="user", content=critique_task_instruction)
    )

    logger.debug(f"Critique request messages prepared for Model 2: {critique_request_messages}")
    return critique_request_messages
