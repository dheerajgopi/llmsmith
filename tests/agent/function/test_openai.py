import unittest
from unittest import mock

from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import (
    Run,
    Message,
    TextContentBlock,
    Text,
    RequiredActionFunctionToolCall,
)
from openai.types.beta.threads.run import (
    RequiredAction,
    RequiredActionSubmitToolOutputs,
)
from openai.types.beta.threads.required_action_function_tool_call import Function
from openai.pagination import AsyncCursorPage
import pytest

from llmsmith.agent.errors import MaxTurnsReachedException
from llmsmith.agent.function.openai import OpenAIFunctionAgent
from llmsmith.agent.tool.openai import OpenAIAssistantTool
from llmsmith.task.models import TaskInput


class OpenAIFunctionAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_with_invalid_input_value(self):
        mock_client = mock.AsyncMock()
        assistant = Assistant(
            id="assistant-001",
            created_at=1,
            model="gpt-4-turbo",
            object="assistant",
            tools=[],
        )
        mock.patch(
            "llmsmith.agent.function.openai.openai.AsyncOpenAI",
            side_effect=mock_client,
        )

        mock_client.beta.assistants.create.return_value = assistant

        agent_task = await OpenAIFunctionAgent.create(
            name="test", llm=mock_client, assistant_options=None, tools=[], max_turns=5
        )

        with pytest.raises(ValueError):
            await agent_task.execute(TaskInput(123))

        assert not mock_client.beta.threads.create.called

    async def test_execute_for_no_function_call_in_llm_response(self):
        mock_client = mock.AsyncMock()
        assistant = Assistant(
            id="assistant-001",
            created_at=1,
            model="gpt-4-turbo",
            object="assistant",
            tools=[],
        )
        thread = Thread(id="thread-001", created_at=1, object="thread")
        run = Run(
            id="run-001",
            assistant_id=assistant.id,
            created_at=1,
            instructions="you are a helpful assistant",
            model="gpt-4-turbo",
            object="thread.run",
            status="completed",
            thread_id=thread.id,
            tools=[],
        )
        msg = Message(
            id="msg-001",
            assistant_id=assistant.id,
            created_at=1,
            content=[
                TextContentBlock(text=Text(annotations=[], value="hello"), type="text")
            ],
            object="thread.message",
            role="assistant",
            run_id=run.id,
            thread_id=thread.id,
            status="completed",
        )
        mock.patch(
            "llmsmith.agent.function.openai.openai.AsyncOpenAI",
            side_effect=mock_client,
        )

        mock_client.beta.assistants.create.return_value = assistant
        mock_client.beta.threads.create.return_value = thread
        mock_client.beta.threads.runs.create_and_poll.return_value = run
        mock_client.beta.threads.messages.list.return_value = AsyncCursorPage(
            data=[msg]
        )

        agent_task = await OpenAIFunctionAgent.create(
            name="test", llm=mock_client, assistant_options=None, tools=[], max_turns=5
        )

        output = await agent_task.execute(TaskInput("query"))

        assert output.content == "hello"

        mock_client.beta.threads.messages.create.assert_called_with(
            thread_id=thread.id, role="user", content="query"
        )

        mock_client.beta.threads.runs.create_and_poll.assert_called_with(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        mock_client.beta.threads.messages.list.assert_called_with(
            thread_id=thread.id,
            run_id=run.id,
            order="desc",
            limit=1,
        )

    async def test_execute_for_max_turns_reached(self):
        mock_client = mock.AsyncMock()
        assistant = Assistant(
            id="assistant-001",
            created_at=1,
            model="gpt-4-turbo",
            object="assistant",
            tools=[],
        )
        thread = Thread(id="thread-001", created_at=1, object="thread")

        res_generator = openai_run_response(assistant, thread)

        def mock_res(**_):
            a = next(res_generator)
            return a

        def some_func():
            return 1

        some_tool = OpenAIAssistantTool(
            declaration={
                "function": {
                    "name": "some_func",
                    "description": "Returns some value.",
                },
                "type": "function",
            },
            callable=some_func,
        )

        run = Run(
            id="run-001",
            assistant_id=assistant.id,
            created_at=1,
            instructions="you are a helpful assistant",
            model="gpt-4-turbo",
            object="thread.run",
            status="requires_action",
            thread_id=thread.id,
            tools=[some_tool.declaration],
            required_action=RequiredAction(
                submit_tool_outputs=RequiredActionSubmitToolOutputs(
                    tool_calls=[
                        RequiredActionFunctionToolCall(
                            id="func-001",
                            type="function",
                            function=Function(arguments="{}", name="some_func"),
                        )
                    ]
                ),
                type="submit_tool_outputs",
            ),
        )

        mock_client.beta.assistants.create.return_value = assistant
        mock_client.beta.threads.create.return_value = thread
        mock_client.beta.threads.runs.create_and_poll.return_value = run

        mock_client.beta.threads.runs.submit_tool_outputs_and_poll.side_effect = (
            mock_res
        )

        agent_task = await OpenAIFunctionAgent.create(
            name="test",
            llm=mock_client,
            assistant_options=None,
            tools=[some_tool],
            max_turns=1,
        )

        with pytest.raises(MaxTurnsReachedException):
            await agent_task.execute(TaskInput("query"))

    async def test_execute_for_function_call_in_llm_response(self):
        mock_client = mock.AsyncMock()
        assistant = Assistant(
            id="assistant-001",
            created_at=1,
            model="gpt-4-turbo",
            object="assistant",
            tools=[],
        )
        thread = Thread(id="thread-001", created_at=1, object="thread")

        res_generator = openai_run_response(assistant, thread)

        def mock_res(**_):
            a = next(res_generator)
            return a

        def some_func():
            return 1

        some_tool = OpenAIAssistantTool(
            declaration={
                "function": {
                    "name": "some_func",
                    "description": "Returns some value.",
                },
                "type": "function",
            },
            callable=some_func,
        )

        run = Run(
            id="run-001",
            assistant_id=assistant.id,
            created_at=1,
            instructions="you are a helpful assistant",
            model="gpt-4-turbo",
            object="thread.run",
            status="requires_action",
            thread_id=thread.id,
            tools=[some_tool.declaration],
            required_action=RequiredAction(
                submit_tool_outputs=RequiredActionSubmitToolOutputs(
                    tool_calls=[
                        RequiredActionFunctionToolCall(
                            id="func-001",
                            type="function",
                            function=Function(arguments="{}", name="some_func"),
                        )
                    ]
                ),
                type="submit_tool_outputs",
            ),
        )
        msg = Message(
            id="msg-001",
            assistant_id=assistant.id,
            created_at=1,
            content=[
                TextContentBlock(text=Text(annotations=[], value="hello"), type="text")
            ],
            object="thread.message",
            role="assistant",
            run_id=run.id,
            thread_id=thread.id,
            status="completed",
        )

        mock_client.beta.assistants.create.return_value = assistant
        mock_client.beta.threads.create.return_value = thread
        mock_client.beta.threads.runs.create_and_poll.return_value = run
        mock_client.beta.threads.messages.list.return_value = AsyncCursorPage(
            data=[msg]
        )

        mock_client.beta.threads.runs.submit_tool_outputs_and_poll.side_effect = (
            mock_res
        )

        agent_task = await OpenAIFunctionAgent.create(
            name="test",
            llm=mock_client,
            assistant_options=None,
            tools=[some_tool],
            max_turns=2,
        )

        output = await agent_task.execute(TaskInput("query"))

        assert output.content == "hello"


def openai_run_response(assistant: Assistant, thread: Thread):
    """Generator for yielding dummy responses in loop"""

    run = Run(
        id="run-002",
        assistant_id=assistant.id,
        created_at=1,
        instructions="you are a helpful assistant",
        model="gpt-4-turbo",
        object="thread.run",
        status="completed",
        thread_id=thread.id,
        tools=[],
    )

    for res in [run]:
        yield res
