import unittest
from unittest import mock

from groq.types.chat.chat_completion import (
    ChatCompletion,
    Choice,
    ChoiceMessage,
    ChoiceLogprobs,
)
import pytest

from llmsmith.task.models import TaskInput
from llmsmith.task.textgen.errors import TextGenFailedException
from llmsmith.task.textgen.groq import GroqTextGenTask
from llmsmith.task.textgen.options.groq import GroqTextGenOptions


class GroqTextGenTaskTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_with_invalid_input_value(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.groq.groq.AsyncGroq",
            side_effect=mock_client,
        )

        text_gen_task = GroqTextGenTask(
            name="test",
            llm=mock_client,
        )

        with pytest.raises(ValueError):
            await text_gen_task.execute(TaskInput(123))

        assert not mock_client.chat.completions.create.called

    async def test_execute_for_no_natural_stop_point_in_response(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.groq.groq.AsyncGroq",
            side_effect=mock_client,
        )

        mock_client.chat.completions.create.return_value = ChatCompletion(
            id="1",
            choices=[
                Choice(
                    index=1,
                    finish_reason="length",
                    logprobs=ChoiceLogprobs(content=None),
                    message=ChoiceMessage(content="hello", role="assistant"),
                )
            ],
            created=1,
            model="llama3-70b-8192",
            object="chat.completion",
        )
        text_gen_task = GroqTextGenTask(
            name="test",
            llm=mock_client,
        )

        with pytest.raises(TextGenFailedException) as err:
            await text_gen_task.execute(TaskInput("query"))

        assert err.value.failure_reason == "NO_NATURAL_STOP_POINT"

        mock_client.chat.completions.create.assert_called_with(
            messages=[{"role": "user", "content": "query"}],
            tools=None,
            model="llama3-70b-8192",
            frequency_penalty=None,
            logit_bias=None,
            logprobs=None,
            max_tokens=None,
            presence_penalty=None,
            response_format=None,
            seed=None,
            stop=None,
            temperature=0.3,
            tool_choice="auto",
            top_logprobs=None,
            top_p=None,
            user=None,
            timeout=None,
        )

    async def test_execute_with_default_llm_options(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.groq.groq.AsyncGroq",
            side_effect=mock_client,
        )

        mock_client.chat.completions.create.return_value = ChatCompletion(
            id="1",
            choices=[
                Choice(
                    index=1,
                    finish_reason="stop",
                    logprobs=ChoiceLogprobs(content=None),
                    message=ChoiceMessage(content="hello", role="assistant"),
                )
            ],
            created=1,
            model="llama3-70b-8192",
            object="chat.completion",
        )
        text_gen_task = GroqTextGenTask(
            name="test",
            llm=mock_client,
        )

        output = await text_gen_task.execute(TaskInput("query"))

        mock_client.chat.completions.create.assert_called_with(
            messages=[{"role": "user", "content": "query"}],
            tools=None,
            model="llama3-70b-8192",
            frequency_penalty=None,
            logit_bias=None,
            logprobs=None,
            max_tokens=None,
            presence_penalty=None,
            response_format=None,
            seed=None,
            stop=None,
            temperature=0.3,
            tool_choice="auto",
            top_logprobs=None,
            top_p=None,
            user=None,
            timeout=None,
        )

        assert output.content == "hello"

    async def test_execute_with_modified_llm_options(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.groq.groq.AsyncGroq",
            side_effect=mock_client,
        )

        mock_client.chat.completions.create.return_value = ChatCompletion(
            id="1",
            choices=[
                Choice(
                    index=1,
                    finish_reason="stop",
                    logprobs=ChoiceLogprobs(content=None),
                    message=ChoiceMessage(content="hello", role="assistant"),
                )
            ],
            created=1,
            model="llama3-70b-8192",
            object="chat.completion",
        )
        text_gen_task = GroqTextGenTask(
            name="test",
            llm=mock_client,
            llm_options=GroqTextGenOptions(
                system_prompt="sys prompt", temperature=0.7, model="test-gpt"
            ),
        )

        output = await text_gen_task.execute(TaskInput("query"))

        mock_client.chat.completions.create.assert_called_with(
            messages=[
                {"role": "user", "content": "query"},
                {"role": "system", "content": "sys prompt"},
            ],
            tools=None,
            model="test-gpt",
            frequency_penalty=None,
            logit_bias=None,
            logprobs=None,
            max_tokens=None,
            presence_penalty=None,
            response_format=None,
            seed=None,
            stop=None,
            temperature=0.7,
            tool_choice="auto",
            top_logprobs=None,
            top_p=None,
            user=None,
            timeout=None,
        )

        assert output.content == "hello"
