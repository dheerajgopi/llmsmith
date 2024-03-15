import logging
from typing import List

try:
    import openai
    from openai.types.chat.chat_completion import ChatCompletion
except ImportError:
    raise ImportError(
        "The 'openai' library is required to use OpenAITextGenTask. You can install it with `pip install llmsmith[openai]`"
    )

from llmsmith.task.base import Task
from llmsmith.task.models import TaskInput, TaskOutput
from llmsmith.task.textgen.options.openai import (
    OpenAITextGenOptions,
    _completion_create_options_dict,
)


log = logging.getLogger(__name__)

# Default options for text generation using OpenAI's LLMs.
default_options: OpenAITextGenOptions = OpenAITextGenOptions(
    model="gpt-3.5-turbo", temperature=0.3
)


class OpenAITextGenTask(Task[str, str]):
    """
    Task for generating text using OpenAI's Large Language Models (LLMs).

    :param name: The name of the task.
    :type name: str
    :param llm: An instance of the Async OpenAI client.
    :type llm: :class:`openai.AsyncOpenAI`
    :param llm_options: A dictionary of options to pass to the OpenAI LLM.
    :type llm_options: :class:`llmsmith.task.textgen.options.openai.OpenAITextGenOptions`, optional
    :raises ValueError: If the name is empty.
    """

    def __init__(
        self,
        name: str,
        llm: openai.AsyncOpenAI,
        llm_options: OpenAITextGenOptions = default_options,
    ) -> None:
        super().__init__(name)

        self.llm: openai.AsyncOpenAI = llm
        self.llm_options: OpenAITextGenOptions = llm_options

    async def execute(self, task_input: TaskInput[str]) -> TaskOutput[str]:
        """
        Generates text using OpenAI LLM using the given input.

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

        sys_prompt = (self.llm_options.get("system_prompt") or "").strip()
        messages_payload: List[dict] = []

        if sys_prompt:
            messages_payload.append({"role": "system", "content": sys_prompt})
        messages_payload.append({"role": "user", "content": llm_input_content})

        chat_completion_options: dict = _completion_create_options_dict(
            self.llm_options
        )

        log.debug(
            f"OpenAI chat request: PAYLOAD: {messages_payload}\n OPTIONS: {chat_completion_options}"
        )

        llm_reply: ChatCompletion = await self.llm.chat.completions.create(
            messages=messages_payload, **chat_completion_options
        )

        log.debug(f"OpenAI chat response: {llm_reply}")

        output_content: str = llm_reply.choices[0].message.content

        log.debug(f"task_output value: {output_content}")

        return TaskOutput(content=output_content, raw_output=llm_reply)
