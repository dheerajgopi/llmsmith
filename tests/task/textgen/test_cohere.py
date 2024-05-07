import unittest
from unittest import mock

from cohere import NonStreamedChatResponse
import pytest

from llmsmith.task.models import TaskInput
from llmsmith.task.textgen.errors import TextGenFailedException
from llmsmith.task.textgen.cohere import CohereTextGenTask
from llmsmith.task.textgen.options.cohere import CohereTextGenOptions


class CohereTextGenTaskTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_with_invalid_input_value(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.cohere.cohere.AsyncClient",
            side_effect=mock_client,
        )

        text_gen_task = CohereTextGenTask(
            name="test",
            llm=mock_client,
        )

        with pytest.raises(ValueError):
            await text_gen_task.execute(TaskInput(123))

        assert not mock_client.chat.called

    async def test_execute_for_failed_safety_check(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.cohere.cohere.AsyncClient",
            side_effect=mock_client,
        )

        mock_client.chat.return_value = NonStreamedChatResponse(
            text="", finish_reason="ERROR_TOXIC"
        )

        text_gen_task = CohereTextGenTask(
            name="test",
            llm=mock_client,
        )

        with pytest.raises(TextGenFailedException) as err:
            await text_gen_task.execute(TaskInput("query"))

        assert err.value.failure_reason == "SAFETY_CHECK_FAILED"

        mock_client.chat.assert_called_with(
            message="query",
            chat_history=None,
            conversation_id=None,
            tools=None,
            tool_results=None,
            model="command-r-plus",
            temperature=0.3,
        )

    async def test_execute_for_limit_exceeded(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.cohere.cohere.AsyncClient",
            side_effect=mock_client,
        )

        mock_client.chat.return_value = NonStreamedChatResponse(
            text="", finish_reason="ERROR_LIMIT"
        )

        text_gen_task = CohereTextGenTask(
            name="test",
            llm=mock_client,
        )

        with pytest.raises(TextGenFailedException) as err:
            await text_gen_task.execute(TaskInput("query"))

        assert err.value.failure_reason == "LIMIT_EXCEEDED"

        mock_client.chat.assert_called_with(
            message="query",
            chat_history=None,
            conversation_id=None,
            tools=None,
            tool_results=None,
            model="command-r-plus",
            temperature=0.3,
        )

    async def test_execute_for_max_tokens(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.cohere.cohere.AsyncClient",
            side_effect=mock_client,
        )

        mock_client.chat.return_value = NonStreamedChatResponse(
            text="", finish_reason="MAX_TOKENS"
        )

        text_gen_task = CohereTextGenTask(
            name="test",
            llm=mock_client,
        )

        with pytest.raises(TextGenFailedException) as err:
            await text_gen_task.execute(TaskInput("query"))

        assert err.value.failure_reason == "MAX_TOKENS_REACHED"

        mock_client.chat.assert_called_with(
            message="query",
            chat_history=None,
            conversation_id=None,
            tools=None,
            tool_results=None,
            model="command-r-plus",
            temperature=0.3,
        )

    async def test_execute_for_cancelled_llm_call(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.cohere.cohere.AsyncClient",
            side_effect=mock_client,
        )

        mock_client.chat.return_value = NonStreamedChatResponse(
            text="", finish_reason="USER_CANCEL"
        )

        text_gen_task = CohereTextGenTask(
            name="test",
            llm=mock_client,
        )

        with pytest.raises(TextGenFailedException) as err:
            await text_gen_task.execute(TaskInput("query"))

        assert err.value.failure_reason == "CANCELLED"

        mock_client.chat.assert_called_with(
            message="query",
            chat_history=None,
            conversation_id=None,
            tools=None,
            tool_results=None,
            model="command-r-plus",
            temperature=0.3,
        )

    async def test_execute_with_default_llm_options(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.cohere.cohere.AsyncClient",
            side_effect=mock_client,
        )

        mock_client.chat.return_value = NonStreamedChatResponse(
            text="llm response", finish_reason="COMPLETE"
        )

        text_gen_task = CohereTextGenTask(
            name="test",
            llm=mock_client,
        )

        output = await text_gen_task.execute(TaskInput("query"))

        mock_client.chat.assert_called_with(
            message="query",
            chat_history=None,
            conversation_id=None,
            tools=None,
            tool_results=None,
            model="command-r-plus",
            temperature=0.3,
        )

        assert output.content == "llm response"

    async def test_execute_with_modified_llm_options(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.cohere.cohere.AsyncClient",
            side_effect=mock_client,
        )

        mock_client.chat.return_value = NonStreamedChatResponse(
            text="llm response", finish_reason="COMPLETE"
        )

        text_gen_task = CohereTextGenTask(
            name="test",
            llm=mock_client,
            llm_options=CohereTextGenOptions(
                system_prompt="sys prompt", temperature=0.7, model="command-r-test"
            ),
        )

        output = await text_gen_task.execute(TaskInput("query"))

        mock_client.chat.assert_called_with(
            message="query",
            chat_history=None,
            conversation_id=None,
            tools=None,
            tool_results=None,
            model="command-r-test",
            temperature=0.7,
            preamble="sys prompt",
        )

        assert output.content == "llm response"
