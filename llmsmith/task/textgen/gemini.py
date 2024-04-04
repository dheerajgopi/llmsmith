import logging
from typing import List

from llmsmith.task.textgen.errors import PromptBlockedError, TextGenFailedError

try:
    from google.generativeai import GenerativeModel
    from google.generativeai.types import GenerateContentResponse
except ImportError:
    raise ImportError(
        "The 'google.generativeai' library is required to use GeminiTextGenTask. You can install it with `pip install \"llmsmith[gemini]\"`"
    )

from llmsmith.task.base import Task
from llmsmith.task.models import TaskInput, TaskOutput
from llmsmith.task.textgen.options.gemini import (
    GeminiTextGenOptions,
    _completion_create_options_dict,
)


log = logging.getLogger(__name__)

# Default options for text generation using Google's Gemini LLMs.
default_options: GeminiTextGenOptions = GeminiTextGenOptions()


class GeminiTextGenTask(Task[str, str]):
    """
    Task for generating text using Google's Gemini Large Language Models (LLMs).

    :param name: The name of the task.
    :type name: str
    :param llm: An instance of the Gemini client.
    :type llm: :class:`google.generativeai.GenerativeModel`
    :param llm_options: A dictionary of options to pass to the Gemini LLM.
    :type llm_options: :class:`llmsmith.task.textgen.options.gemini.GeminiTextGenOptions`, optional
    :raises ValueError: If the name is empty.
    """

    def __init__(
        self,
        name: str,
        llm: GenerativeModel,
        llm_options: GeminiTextGenOptions = default_options,
    ) -> None:
        super().__init__(name)

        self.llm: GenerativeModel = llm
        self.llm_options: GeminiTextGenOptions = llm_options

    async def execute(self, task_input: TaskInput[str]) -> TaskOutput[str]:
        """
        Generates text using Gemini LLM using the given input.

        :param task_input: The input to the task.
        :type task_input: :class:`llmsmith.task.models.TaskInput[str]`
        :raises ValueError: If the content of the task input is not a string.
        :raises PromptBlockedError: If the prompt is blocked by the AI.
        :raises TextGenFailedError: If AI fails to generate text based on the prompt.
        :returns: The output of the task.
        :rtype: :class:`llmsmith.task.models.TaskOutput[str]`
        """
        if not isinstance(task_input.content, str):
            log.debug(f"task_input value: {task_input}")
            raise ValueError("task_input.content should be of type 'str'")

        llm_input_content: str = task_input.content

        messages_payload: List[dict] = [{"role": "user", "parts": [llm_input_content]}]
        chat_completion_options: dict = _completion_create_options_dict(
            self.llm_options
        )

        log.debug(
            f"Google Gemini chat request: PAYLOAD: {messages_payload}\n OPTIONS: {chat_completion_options}"
        )

        llm_reply: GenerateContentResponse = await self.llm.generate_content_async(
            contents=messages_payload, **chat_completion_options
        )

        log.info(f"Google Gemini chat response: {llm_reply}")

        if (
            llm_reply.prompt_feedback.block_reason
            and llm_reply.prompt_feedback.block_reason
            != llm_reply.prompt_feedback.BlockReason.BLOCK_REASON_UNSPECIFIED
        ):
            raise PromptBlockedError(
                "Prompt blocked by the AI",
                block_reason=GeminiTextGenTask._block_reason_str(llm_reply),
            )

        output_candidate = next(
            (c for c in llm_reply.candidates if c.finish_reason == c.FinishReason.STOP),
            None,
        )
        if not output_candidate:
            raise TextGenFailedError(
                "Failed to generate text", failure_reason="NO_NATURAL_STOP_POINT"
            )

        output_content: str = next(
            (p.text for p in output_candidate.content.parts if p.text), None
        )
        if not output_content:
            raise TextGenFailedError(
                "Failed to generate text", failure_reason="NO_TEXT_DATA"
            )

        log.debug(f"task_output value: {output_content}")

        return TaskOutput(content=output_content, raw_output=llm_reply)

    @classmethod
    def _block_reason_str(llm_reply: GenerateContentResponse) -> str:
        if (
            llm_reply.prompt_feedback.block_reason
            == llm_reply.prompt_feedback.BlockReason.SAFETY
        ):
            return "SAFETY_CHECK_FAILED"

        return "OTHER"
