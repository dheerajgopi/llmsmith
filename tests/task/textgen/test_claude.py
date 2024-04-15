import unittest
from unittest import mock

from anthropic.types.message import Message
from anthropic.types.content_block import ContentBlock
import pytest

from llmsmith.task.models import TaskInput
from llmsmith.task.textgen.claude import ClaudeTextGenTask
from llmsmith.task.textgen.options.claude import ClaudeTextGenOptions


class ClaudeTextGenTaskTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_with_invalid_input_value(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.claude.anthropic.AsyncAnthropic",
            side_effect=mock_client,
        )

        text_gen_task = ClaudeTextGenTask(
            name="test",
            llm=mock_client,
        )

        with pytest.raises(ValueError):
            await text_gen_task.execute(TaskInput(123))

        assert not mock_client.messages.create.called

    async def test_execute_with_default_llm_options(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.claude.anthropic.AsyncAnthropic",
            side_effect=mock_client,
        )

        mock_client.messages.create.return_value = Message(
            id="1",
            content=[ContentBlock(type="text", text="hello")],
            model="claude-3-opus-20240229",
            role="assistant",
            type="message",
            usage={"input_tokens": 1, "output_tokens": 1},
        )
        text_gen_task = ClaudeTextGenTask(
            name="test",
            llm=mock_client,
        )

        output = await text_gen_task.execute(TaskInput("query"))

        mock_client.messages.create.assert_called_with(
            messages=[{"role": "user", "content": "query"}],
            model="claude-3-opus-20240229",
            max_tokens=1024,
            temperature=0.3,
        )

        assert output.content == "hello"

    async def test_execute_with_modified_llm_options(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.claude.anthropic.AsyncAnthropic",
            side_effect=mock_client,
        )

        mock_client.messages.create.return_value = Message(
            id="1",
            content=[ContentBlock(type="text", text="hello")],
            model="claude-3-opus-20240229",
            role="assistant",
            type="message",
            usage={"input_tokens": 1, "output_tokens": 1},
        )
        text_gen_task = ClaudeTextGenTask(
            name="test",
            llm=mock_client,
            llm_options=ClaudeTextGenOptions(
                system="sys prompt", temperature=0.7, model="test-claude"
            ),
        )

        output = await text_gen_task.execute(TaskInput("query"))

        mock_client.messages.create.assert_called_with(
            messages=[{"role": "user", "content": "query"}],
            model="test-claude",
            system="sys prompt",
            max_tokens=1024,
            temperature=0.7,
        )

        assert output.content == "hello"
