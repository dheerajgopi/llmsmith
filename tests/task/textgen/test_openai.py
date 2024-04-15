import unittest
from unittest import mock

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import pytest

from llmsmith.task.models import TaskInput
from llmsmith.task.textgen.openai import OpenAITextGenTask
from llmsmith.task.textgen.options.openai import OpenAITextGenOptions


class OpenAITextGenTaskTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_with_invalid_input_value(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.openai.openai.AsyncOpenAI",
            side_effect=mock_client,
        )

        text_gen_task = OpenAITextGenTask(
            name="test",
            llm=mock_client,
        )

        with pytest.raises(ValueError):
            await text_gen_task.execute(TaskInput(123))

        assert not mock_client.chat.completions.create.called

    async def test_execute_with_default_llm_options(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.openai.openai.AsyncOpenAI",
            side_effect=mock_client,
        )

        mock_client.chat.completions.create.return_value = ChatCompletion(
            id="1",
            choices=[
                Choice(
                    index=1,
                    finish_reason="stop",
                    message=ChatCompletionMessage(content="hello", role="assistant"),
                )
            ],
            created=1,
            model="gpt-3.5-turbo",
            object="chat.completion",
        )
        text_gen_task = OpenAITextGenTask(
            name="test",
            llm=mock_client,
        )

        output = await text_gen_task.execute(TaskInput("query"))

        mock_client.chat.completions.create.assert_called_with(
            messages=[{"role": "user", "content": "query"}],
            model="gpt-3.5-turbo",
            frequency_penalty=None,
            function_call=None,
            functions=None,
            logit_bias=None,
            logprobs=None,
            max_tokens=None,
            presence_penalty=None,
            response_format=None,
            seed=None,
            stop=None,
            temperature=0.3,
            tool_choice=None,
            tools=None,
            top_logprobs=None,
            top_p=None,
            user=None,
            timeout=None,
        )

        assert output.content == "hello"

    async def test_execute_with_modified_llm_options(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.openai.openai.AsyncOpenAI",
            side_effect=mock_client,
        )

        mock_client.chat.completions.create.return_value = ChatCompletion(
            id="1",
            choices=[
                Choice(
                    index=1,
                    finish_reason="stop",
                    message=ChatCompletionMessage(content="hello", role="assistant"),
                )
            ],
            created=1,
            model="gpt-3.5-turbo",
            object="chat.completion",
        )
        text_gen_task = OpenAITextGenTask(
            name="test",
            llm=mock_client,
            llm_options=OpenAITextGenOptions(
                system_prompt="sys prompt", temperature=0.7, model="test-gpt"
            ),
        )

        output = await text_gen_task.execute(TaskInput("query"))

        mock_client.chat.completions.create.assert_called_with(
            messages=[
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "query"},
            ],
            model="test-gpt",
            frequency_penalty=None,
            function_call=None,
            functions=None,
            logit_bias=None,
            logprobs=None,
            max_tokens=None,
            presence_penalty=None,
            response_format=None,
            seed=None,
            stop=None,
            temperature=0.7,
            tool_choice=None,
            tools=None,
            top_logprobs=None,
            top_p=None,
            user=None,
            timeout=None,
        )

        assert output.content == "hello"
