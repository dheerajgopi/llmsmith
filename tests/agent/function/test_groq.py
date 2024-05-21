import unittest
from unittest import mock

from groq.types.chat.chat_completion import (
    ChatCompletion,
    Choice,
    ChoiceMessage,
    ChoiceLogprobs,
    ChoiceMessageToolCall,
    ChoiceMessageToolCallFunction,
)
import pytest

from llmsmith.agent.errors import MaxTurnsReachedException
from llmsmith.agent.function.groq import GroqFunctionAgent
from llmsmith.agent.tool.groq import GroqTool
from llmsmith.task.models import TaskInput


class GroqFunctionAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_with_invalid_input_value(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.groq.groq.AsyncGroq",
            side_effect=mock_client,
        )

        agent_task = GroqFunctionAgent(
            name="test", llm=mock_client, llm_options=None, max_turns=5
        )

        with pytest.raises(ValueError):
            await agent_task.execute(TaskInput(123))

        assert not mock_client.chat.completions.create.called

    async def test_execute_for_no_function_call_in_llm_response(self):
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
                    message=ChoiceMessage(content="llm response", role="assistant"),
                )
            ],
            created=1,
            model="llama3-70b-8192",
            object="chat.completion",
        )
        text_gen_task = GroqFunctionAgent(
            name="test", llm=mock_client, llm_options=None, max_turns=5
        )

        output = await text_gen_task.execute(TaskInput("query"))

        assert output.content == "llm response"
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

    async def test_execute_for_max_turns_reached(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.groq.groq.AsyncGroq",
            side_effect=mock_client,
        )

        res_generator = groq_response_with_function_call()

        def mock_res(**_):
            a = next(res_generator)
            return a

        def some_func():
            return 1

        mock_client.chat.completions.create.side_effect = mock_res
        text_gen_task = GroqFunctionAgent(
            name="test",
            llm=mock_client,
            llm_options=None,
            tools=[
                GroqTool(
                    declaration={
                        "function": {
                            "name": "some_func",
                            "description": "Returns the result of some function.",
                        },
                        "type": "function",
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
            "llmsmith.task.textgen.groq.groq.AsyncGroq",
            side_effect=mock_client,
        )

        res_generator = groq_response_with_function_call()

        def mock_res(**_):
            a = next(res_generator)
            return a

        def some_func():
            return 1

        mock_client.chat.completions.create.side_effect = mock_res
        text_gen_task = GroqFunctionAgent(
            name="test",
            llm=mock_client,
            llm_options=None,
            tools=[
                GroqTool(
                    declaration={
                        "function": {
                            "name": "some_func",
                            "description": "Returns the result of some function.",
                        },
                        "type": "function",
                    },
                    callable=some_func,
                )
            ],
            max_turns=5,
        )

        output = await text_gen_task.execute(TaskInput("query"))

        assert output.content == "llm response"


def groq_response_with_function_call():
    """Generator for yielding dummy responses in loop"""

    agent_res_1_val = ChatCompletion(
        id="1",
        choices=[
            Choice(
                index=1,
                finish_reason="tool_calls",
                logprobs=ChoiceLogprobs(content=None),
                message=ChoiceMessage(
                    content="",
                    role="assistant",
                    tool_calls=[
                        ChoiceMessageToolCall(
                            id="func1",
                            function=ChoiceMessageToolCallFunction(name="some_func"),
                        )
                    ],
                ),
            )
        ],
        created=1,
        model="llama3-70b-8192",
        object="chat.completion",
    )

    agent_res_2_val = ChatCompletion(
        id="1",
        choices=[
            Choice(
                index=1,
                finish_reason="stop",
                logprobs=ChoiceLogprobs(content=None),
                message=ChoiceMessage(content="llm response", role="assistant"),
            )
        ],
        created=1,
        model="llama3-70b-8192",
        object="chat.completion",
    )

    for res in [agent_res_1_val, agent_res_2_val]:
        yield res
