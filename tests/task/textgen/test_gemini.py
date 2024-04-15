import unittest
from unittest import mock

from google.generativeai.types import GenerateContentResponse, HarmBlockThreshold
from google.ai.generativelanguage_v1beta.types.generative_service import (
    GenerateContentResponse as ContentResponse,
)
from google.ai.generativelanguage_v1beta.types.generative_service import Candidate
from google.ai.generativelanguage_v1beta.types.safety import SafetyRating, HarmCategory
from google.ai.generativelanguage_v1beta.types import Content, Part
import pytest

from llmsmith.task.models import TaskInput
from llmsmith.task.textgen.errors import PromptBlockedError, TextGenFailedError
from llmsmith.task.textgen.gemini import GeminiTextGenTask
from llmsmith.task.textgen.options.gemini import GeminiTextGenOptions


class GeminiTextGenTaskTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_with_invalid_input_value(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.gemini.GenerativeModel",
            side_effect=mock_client,
        )

        text_gen_task = GeminiTextGenTask(
            name="test",
            llm=mock_client,
        )

        with pytest.raises(ValueError):
            await text_gen_task.execute(TaskInput(123))

        assert not mock_client.generate_content_async.called

    async def test_execute_for_blocked_prompt(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.gemini.GenerativeModel",
            side_effect=mock_client,
        )

        # Create GenerativeModel response object
        safety_rating = SafetyRating()
        safety_rating.category = HarmCategory(2)
        safety_rating.blocked = True
        safety_rating.probability = safety_rating.HarmProbability.LOW

        prompt_feedback = ContentResponse.PromptFeedback()
        prompt_feedback.block_reason = prompt_feedback.BlockReason.SAFETY
        prompt_feedback.safety_ratings = [safety_rating]

        content_part = Part()
        content_part.text = "blocked"
        content = Content()
        content.parts = [content_part]
        candidate = Candidate()
        candidate.index = 1
        candidate.content = content
        candidate.finish_reason = candidate.FinishReason.SAFETY
        candidate.safety_ratings = [safety_rating]

        content_res = ContentResponse()
        content_res.prompt_feedback = prompt_feedback
        content_res.candidates = [candidate]
        response_val = GenerateContentResponse(
            done=True, iterator=[content_res], result=content_res
        )

        mock_client.generate_content_async.return_value = response_val
        text_gen_task = GeminiTextGenTask(
            name="test",
            llm=mock_client,
        )

        with pytest.raises(PromptBlockedError):
            await text_gen_task.execute(TaskInput("query"))

        mock_client.generate_content_async.assert_called_with(
            contents=[{"role": "user", "parts": ["query"]}],
            generation_config=None,
            safety_settings=None,
            tools=None,
            request_options=None,
        )

    async def test_execute_for_no_natural_stop_point_in_response(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.textgen.gemini.GenerativeModel",
            side_effect=mock_client,
        )

        # Create GenerativeModel response object
        safety_rating = SafetyRating()
        safety_rating.category = HarmCategory(2)
        safety_rating.blocked = True
        safety_rating.probability = safety_rating.HarmProbability.LOW

        prompt_feedback = ContentResponse.PromptFeedback()
        prompt_feedback.block_reason = (
            prompt_feedback.BlockReason.BLOCK_REASON_UNSPECIFIED
        )
        prompt_feedback.safety_ratings = [safety_rating]

        content_part = Part()
        content_part.text = "blocked"
        content = Content()
        content.parts = [content_part]
        candidate = Candidate()
        candidate.index = 1
        candidate.content = content
        candidate.finish_reason = candidate.FinishReason.SAFETY
        candidate.safety_ratings = [safety_rating]

        content_res = ContentResponse()
        content_res.prompt_feedback = prompt_feedback
        content_res.candidates = [candidate]
        response_val = GenerateContentResponse(
            done=True, iterator=[content_res], result=content_res
        )

        mock_client.generate_content_async.return_value = response_val
        text_gen_task = GeminiTextGenTask(
            name="test",
            llm=mock_client,
        )

        with pytest.raises(TextGenFailedError) as err:
            await text_gen_task.execute(TaskInput("query"))

        assert err.value.failure_reason == "NO_NATURAL_STOP_POINT"
        mock_client.generate_content_async.assert_called_with(
            contents=[{"role": "user", "parts": ["query"]}],
            generation_config=None,
            safety_settings=None,
            tools=None,
            request_options=None,
        )

    async def test_execute_for_no_text_data_in_response(self):
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
        content_part.text = None
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
        text_gen_task = GeminiTextGenTask(
            name="test",
            llm=mock_client,
        )

        with pytest.raises(TextGenFailedError) as err:
            await text_gen_task.execute(TaskInput("query"))

        assert err.value.failure_reason == "NO_TEXT_DATA"
        mock_client.generate_content_async.assert_called_with(
            contents=[{"role": "user", "parts": ["query"]}],
            generation_config=None,
            safety_settings=None,
            tools=None,
            request_options=None,
        )

    async def test_execute_with_default_llm_options(self):
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
        text_gen_task = GeminiTextGenTask(
            name="test",
            llm=mock_client,
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

    async def test_execute_with_modified_llm_options(self):
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
        text_gen_task = GeminiTextGenTask(
            name="test",
            llm=mock_client,
            llm_options=GeminiTextGenOptions(
                generation_config={"temperature": 0.5},
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
                },
            ),
        )

        output = await text_gen_task.execute(TaskInput("query"))

        assert output.content == "hello"
        mock_client.generate_content_async.assert_called_with(
            contents=[{"role": "user", "parts": ["query"]}],
            generation_config={"temperature": 0.5},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            },
            tools=None,
            request_options=None,
        )
