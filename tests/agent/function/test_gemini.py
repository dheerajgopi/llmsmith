import unittest
from unittest import mock

from google.generativeai.types import GenerateContentResponse
from google.ai.generativelanguage_v1beta.types.generative_service import (
    GenerateContentResponse as ContentResponse,
)
from google.ai.generativelanguage_v1beta.types.generative_service import Candidate
from google.ai.generativelanguage_v1beta.types.safety import SafetyRating, HarmCategory
from google.ai.generativelanguage_v1beta.types import Content, Part
import pytest

from llmsmith.agent.errors import MaxTurnsReachedException
from llmsmith.agent.function.gemini import GeminiFunctionAgent
from llmsmith.agent.tool.gemini import GeminiTool
from llmsmith.task.models import TaskInput


class GeminiFunctionAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_with_invalid_input_value(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.gemini.GenerativeModel",
            side_effect=mock_client,
        )

        agent_task = GeminiFunctionAgent(
            name="test", llm=mock_client, llm_options=None, max_turns=5
        )

        with pytest.raises(ValueError):
            await agent_task.execute(TaskInput(123))

        assert not mock_client.generate_content_async.called

    async def test_execute_for_no_function_call_in_llm_response(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.gemini.GenerativeModel",
            side_effect=mock_client,
        )

        # Create GenerativeModel response object
        safety_rating = SafetyRating()
        safety_rating.category = HarmCategory(0)
        safety_rating.blocked = False
        safety_rating.probability = (
            safety_rating.HarmProbability.HARM_PROBABILITY_UNSPECIFIED
        )

        prompt_feedback = ContentResponse.PromptFeedback()
        prompt_feedback.block_reason = (
            prompt_feedback.BlockReason.BLOCK_REASON_UNSPECIFIED
        )
        prompt_feedback.safety_ratings = [safety_rating]

        content_part = Part()
        content_part.text = "hello"
        content = Content()
        content.parts = [content_part]
        candidate = Candidate()
        candidate.index = 1
        candidate.content = content
        candidate.finish_reason = candidate.FinishReason.STOP
        candidate.safety_ratings = [safety_rating]

        content_res = ContentResponse()
        content_res.prompt_feedback = prompt_feedback
        content_res.candidates = [candidate]
        response_val = GenerateContentResponse(
            done=True, iterator=[content_res], result=content_res
        )

        mock_client.generate_content_async.return_value = response_val
        text_gen_task = GeminiFunctionAgent(
            name="test", llm=mock_client, llm_options=None, max_turns=5
        )

        output = await text_gen_task.execute(TaskInput("query"))

        assert output.content == "hello"
        mock_client.generate_content_async.assert_called_with(
            contents=[{"role": "user", "parts": ["query"]}],
            generation_config=None,
            safety_settings=None,
            tools=None,
            request_options=None,
        )

    async def test_execute_for_max_turns_reached(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.gemini.GenerativeModel",
            side_effect=mock_client,
        )

        res_generator = gemini_response_with_function_call()

        def mock_res(**_):
            a = next(res_generator)
            return a

        def some_func():
            return 1

        mock_client.generate_content_async.side_effect = mock_res
        text_gen_task = GeminiFunctionAgent(
            name="test",
            llm=mock_client,
            llm_options=None,
            tools=[
                GeminiTool(
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
            "llmsmith.task.textgen.gemini.GenerativeModel",
            side_effect=mock_client,
        )

        res_generator = gemini_response_with_function_call()

        def mock_res(**_):
            a = next(res_generator)
            return a

        def some_func():
            return 1

        mock_client.generate_content_async.side_effect = mock_res
        text_gen_task = GeminiFunctionAgent(
            name="test",
            llm=mock_client,
            llm_options=None,
            tools=[
                GeminiTool(
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

        assert output.content == "The result is something"


def gemini_response_with_function_call():
    """Generator for yielding dummy responses in loop"""

    safety_rating = SafetyRating()
    safety_rating.category = HarmCategory(0)
    safety_rating.blocked = False
    safety_rating.probability = (
        safety_rating.HarmProbability.HARM_PROBABILITY_UNSPECIFIED
    )

    prompt_feedback = ContentResponse.PromptFeedback()
    prompt_feedback.block_reason = prompt_feedback.BlockReason.BLOCK_REASON_UNSPECIFIED
    prompt_feedback.safety_ratings = [safety_rating]

    agent_res_1_part = Part()
    agent_res_1_part.text = None
    agent_res_1_content = Content()
    agent_res_1_content.parts = [{"function_call": {"name": "some_func"}}]
    agent_res_1_candidate = Candidate()
    agent_res_1_candidate.index = 1
    agent_res_1_candidate.content = agent_res_1_content
    agent_res_1_candidate.finish_reason = agent_res_1_candidate.FinishReason.STOP
    agent_res_1_candidate.safety_ratings = [safety_rating]

    agent_res_1 = ContentResponse()
    agent_res_1.prompt_feedback = prompt_feedback
    agent_res_1.candidates = [agent_res_1_candidate]
    agent_res_1_val = GenerateContentResponse(
        done=True, iterator=[agent_res_1], result=agent_res_1
    )

    agent_res_2_part = Part()
    agent_res_2_part.text = "The result is something"
    agent_res_2_content = Content()
    agent_res_2_content.parts = [agent_res_2_part]
    agent_res_2_candidate = Candidate()
    agent_res_2_candidate.index = 1
    agent_res_2_candidate.content = agent_res_2_content
    agent_res_2_candidate.finish_reason = agent_res_2_candidate.FinishReason.STOP
    agent_res_2_candidate.safety_ratings = [safety_rating]

    agent_res_2 = ContentResponse()
    agent_res_2.prompt_feedback = prompt_feedback
    agent_res_2.candidates = [agent_res_2_candidate]
    agent_res_2_val = GenerateContentResponse(
        done=True, iterator=[agent_res_2], result=agent_res_2
    )

    for res in [agent_res_1_val, agent_res_2_val]:
        yield res
