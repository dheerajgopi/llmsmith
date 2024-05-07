import logging
from typing import List

try:
    import anthropic
    from anthropic.types.message import Message
except ImportError:
    raise ImportError(
        "The 'anthropic' library is required to use Claude LLMs. You can install it with `pip install \"llmsmith[claude]\"`"
    )

from llmsmith.task.base import Task
from llmsmith.task.models import TaskInput, TaskOutput
from llmsmith.task.textgen.options.claude import (
    ClaudeTextGenOptions,
    _completion_create_options_dict,
)


log = logging.getLogger(__name__)

# Default options for text generation using Anthropic's Claude LLMs.
default_options: ClaudeTextGenOptions = ClaudeTextGenOptions(
    model="claude-3-opus-20240229", temperature=0.3, max_tokens=1024
)


class ClaudeTextGenTask(Task[str, str]):
    """
    Task for generating text using Anthropic's Claude Large Language Models (LLMs).

    :param name: The name of the task.
    :type name: str
    :param llm: An instance of the Async Anthropic client.
    :type llm: :class:`anthropic.AsyncAnthropic`
    :param llm_options: A dictionary of options to pass to the Anthropic Claude LLM.
    :type llm_options: :class:`llmsmith.task.textgen.options.claude.ClaudeTextGenOptions`, optional
    :raises ValueError: If the name is empty.
    """

    def __init__(
        self,
        name: str,
        llm: anthropic.AsyncAnthropic,
        llm_options: ClaudeTextGenOptions = default_options,
    ) -> None:
        super().__init__(name)

        self.llm: anthropic.AsyncAnthropic = llm
        self.llm_options: ClaudeTextGenOptions = llm_options

    async def execute(self, task_input: TaskInput[str]) -> TaskOutput[str]:
        """
        Generates text using Anthropic Claude LLM using the given input.

        :param task_input: The input to the task.
        :type task_input: :class:`llmsmith.task.models.TaskInput[str]`
        :raises ValueError: If the content of the task input is not a string.
        :returns: The output of the task.
        :rtype: :class:`llmsmith.task.models.TaskOutput[str]`
        """
        if not isinstance(task_input.content, str):
            log.debug(f"task_input value: {task_input}")
            raise ValueError("task_input.content should be of type 'str'")

        llm_input_content: str = task_input.content

        messages_payload: List[dict] = [{"role": "user", "content": llm_input_content}]
        chat_completion_options: dict = _completion_create_options_dict(
            self.llm_options
        )

        log.debug(
            f"Anthropic Claude chat request: PAYLOAD: {messages_payload}\n OPTIONS: {chat_completion_options}"
        )

        llm_reply: Message = await self.llm.messages.create(
            messages=messages_payload, **chat_completion_options
        )

        log.debug(f"Anthropic Claude chat response: {llm_reply}")

        output_content: str = llm_reply.content[0].text

        log.debug(f"task_output value: {output_content}")

        return TaskOutput(content=output_content, raw_output=llm_reply)
