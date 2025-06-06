# Placeholder for critique_service tests
# We will add tests for prepare_critique_messages

import pytest
from src.models.openai_types import CustomChatCompletionMessage
from openai.types.chat import ChatCompletionMessage as OpenAIChatCompletionMessage
from src.services.critique_service import prepare_critique_messages

def test_prepare_critique_messages_basic():
    original_user_messages = [
        CustomChatCompletionMessage(role="user", content="What is Python?")
    ]
    model_1_response = OpenAIChatCompletionMessage(role="assistant", content="Python is a programming language.")
    critique_system_prompt_template = (
        "Critique the following: User: {original_request} Model1: {model_response}"
    )

    critique_messages = prepare_critique_messages(
        original_request_messages=original_user_messages, # Parameter name updated in service
        model1_response_message=model_1_response, # Parameter name updated in service
        critique_system_prompt_template=critique_system_prompt_template
    )

    assert len(critique_messages) == 2
    assert critique_messages[0].role == "system"
    assert "What is Python?" in critique_messages[0].content
    assert "Python is a programming language." in critique_messages[0].content
    assert critique_messages[1].role == "user"
    expected_user_content_for_critique = (
        "Original User Request:\n```\nWhat is Python?\n```\n\n"
        "Model 1 Response to Critique:\n```\nPython is a programming language.\n```"
    )
    assert critique_messages[1].content == expected_user_content_for_critique

# Add more tests for edge cases, multiple user messages, etc.
