import json
import logging
from typing import List, Union

from llmsmith.task.textgen.errors import TextGenFailedException

try:
    import openai
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )
    from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
except ImportError:
    raise ImportError(
        "The 'openai' library is required to use OpenAI LLMs. You can install it with `pip install \"llmsmith[openai]\"`"
    )

from llmsmith.task.base import Task
from llmsmith.task.models import ChatResponse, FunctionCall, TaskInput, TaskOutput
from llmsmith.task.textgen.options.openai import (
    OpenAITextGenOptions,
    _completion_create_options_dict,
)


log = logging.getLogger(__name__)

# Default options for text generation using OpenAI's LLMs.
default_options: OpenAITextGenOptions = OpenAITextGenOptions(
    model="gpt-3.5-turbo", temperature=0.3
)


class BaseOpenAIChat:
    """
    Base class for chatting using OpenAI Large Language Models (LLMs).

    :param llm: An instance of the Async OpenAI client.
    :type llm: :class:`openai.AsyncOpenAI`
    :param llm_options: A dictionary of options to pass to the OpenAI LLM.
    :type llm_options: :class:`llmsmith.task.textgen.options.openai.OpenAITextGenOptions`, optional
    """

    def __init__(
        self,
        llm: openai.AsyncOpenAI,
        llm_options: OpenAITextGenOptions = default_options,
    ) -> None:
        self.llm: openai.AsyncOpenAI = llm
        self.llm_options: OpenAITextGenOptions = llm_options or default_options

    async def chat(
        self,
        messages_payload: List[ChatCompletionMessageParam],
        tools: Union[List[ChatCompletionToolParam], None] = None,
    ) -> ChatResponse:
        """
        Generates text using OpenAI LLM using the given input.

        :param messages_payload: The input messages for the chat.
        :type messages_payload: List[:class:`openai.types.chat.chat_completion_tool_param.ChatCompletionMessageParam`]
        :param tools: Tools (functions) which can be used by the LLM.
        :type tools: List[:class:`openai.types.chat.chat_completion_tool_param.ChatCompletionMessageParam`], optional
        :raises TextGenFailedError: If AI fails to generate text based on the prompt.
        :returns: chat response from the LLM.
        :rtype: :class:`llmsmith.task.models.ChatResponse`
        """
        sys_prompt = (self.llm_options.get("system_prompt") or "").strip()
        sys_prompt_in_payload = next(
            (msg for msg in messages_payload if msg.get("role") == "system"), None
        )

        # Add system prompt if provided in llm options and not available in the messages payload
        if sys_prompt and not sys_prompt_in_payload:
            messages_payload.append({"role": "system", "content": sys_prompt})

        chat_completion_options: dict = _completion_create_options_dict(
            self.llm_options
        )

        log.debug(
            f"OpenAI chat request: PAYLOAD: {messages_payload}\n OPTIONS: {chat_completion_options}"
        )

        llm_reply: ChatCompletion = await self.llm.chat.completions.create(
            messages=messages_payload, tools=tools, **chat_completion_options
        )

        log.debug(f"OpenAI chat response: {llm_reply}")

        output_choice_with_func_call = next(
            (c for c in llm_reply.choices if c.finish_reason == "tool_calls"),
            None,
        )

        if output_choice_with_func_call:
            return ChatResponse(
                text=output_choice_with_func_call.message.content,
                raw_output=llm_reply,
                function_calls={
                    tool.function.name: FunctionCall(
                        id=tool.id,
                        args=(
                            json.loads(tool.function.arguments)
                            if tool.function.arguments
                            else {}
                        ),
                    )
                    for tool in output_choice_with_func_call.message.tool_calls
                },
            )

        output_choice = next(
            (c for c in llm_reply.choices if c.finish_reason == "stop"),
            None,
        )

        if not output_choice:
            raise TextGenFailedException(
                "Failed to generate text", failure_reason="NO_NATURAL_STOP_POINT"
            )

        output_content: str = output_choice.message.content

        log.debug(f"chat response output value: {output_content}")

        return ChatResponse(text=output_content, raw_output=llm_reply)


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
        self._chat = BaseOpenAIChat(llm, llm_options)

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
        messages_payload: List[dict] = [{"role": "user", "content": llm_input_content}]

        chat_response = await self._chat.chat(messages_payload)

        return TaskOutput(
            content=chat_response.text, raw_output=chat_response.raw_output
        )
