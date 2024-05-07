import unittest
from unittest import mock

from cohere import NonStreamedChatResponse, ToolCall
import pytest

from llmsmith.agent.errors import MaxTurnsReachedException
from llmsmith.agent.function.cohere import CohereFunctionAgent
from llmsmith.agent.tool.cohere import CohereTool
from llmsmith.task.models import TaskInput


class CohereFunctionAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_with_invalid_input_value(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.cohere.cohere.AsyncClient",
            side_effect=mock_client,
        )

        agent_task = CohereFunctionAgent(
            name="test", llm=mock_client, llm_options=None, max_turns=5
        )

        with pytest.raises(ValueError):
            await agent_task.execute(TaskInput(123))

        assert not mock_client.chat.called

    async def test_execute_for_no_function_call_in_llm_response(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.cohere.cohere.AsyncClient",
            side_effect=mock_client,
        )

        mock_client.chat.return_value = NonStreamedChatResponse(
            text="llm response", finish_reason="COMPLETE"
        )
        text_gen_task = CohereFunctionAgent(
            name="test", llm=mock_client, llm_options=None, max_turns=5
        )

        output = await text_gen_task.execute(TaskInput("query"))

        assert output.content == "llm response"
        mock_client.chat.assert_called_with(
            message="query",
            chat_history=None,
            conversation_id=None,
            tools=[],
            tool_results=None,
            model="command-r-plus",
            temperature=0.3,
        )

    async def test_execute_for_max_turns_reached(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.cohere.cohere.AsyncClient",
            side_effect=mock_client,
        )

        res_generator = cohere_response_with_function_call()

        def mock_res(**_):
            a = next(res_generator)
            return a

        def some_func():
            return 1

        mock_client.chat.side_effect = mock_res
        text_gen_task = CohereFunctionAgent(
            name="test",
            llm=mock_client,
            llm_options=None,
            tools=[
                CohereTool(
                    declaration={
                        "name": "some_func",
                        "description": "Returns the result of some function.",
                    },
                    callable=some_func,
                )
            ],
            max_turns=1,
        )

        with pytest.raises(MaxTurnsReachedException):
            await text_gen_task.execute(TaskInput("query"))

    async def test_execute_for_function_call_in_llm_response(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.cohere.cohere.AsyncClient",
            side_effect=mock_client,
        )

        res_generator = cohere_response_with_function_call()

        def mock_res(**_):
            a = next(res_generator)
            return a

        def some_func():
            return 1

        mock_client.chat.side_effect = mock_res
        text_gen_task = CohereFunctionAgent(
            name="test",
            llm=mock_client,
            llm_options=None,
            tools=[
                CohereTool(
                    declaration={
                        "name": "some_func",
                        "description": "Returns the result of some function.",
                    },
                    callable=some_func,
                )
            ],
            max_turns=5,
        )

        output = await text_gen_task.execute(TaskInput("query"))

        assert output.content == "llm response"


def cohere_response_with_function_call():
    """Generator for yielding dummy responses in loop"""

    agent_res_1_val = NonStreamedChatResponse(
        text="",
        finish_reason="COMPLETE",
        tool_calls=[ToolCall(name="some_func", parameters={})],
    )

    agent_res_2_val = NonStreamedChatResponse(
        text="llm response", finish_reason="COMPLETE", tool_calls=[]
    )

    for res in [agent_res_1_val, agent_res_2_val]:
        yield res
