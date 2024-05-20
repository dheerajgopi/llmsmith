import json
import logging
from typing import List, Union

from llmsmith.task.textgen.errors import TextGenFailedException

try:
    import groq
    from groq.types.chat.chat_completion import ChatCompletion
    from groq.types.chat.completion_create_params import (
        Message,
        Tool,
    )
except ImportError:
    raise ImportError(
        "The 'groq' library is required to use LLMs in Groq. You can install it with `pip install \"llmsmith[groq]\"`"
    )

from llmsmith.task.base import Task
from llmsmith.task.models import ChatResponse, FunctionCall, TaskInput, TaskOutput
from llmsmith.task.textgen.options.groq import (
    GroqTextGenOptions,
    _completion_create_options_dict,
)


log = logging.getLogger(__name__)

# Default options for text generation using LLMs in Groq Cloud.
default_options: GroqTextGenOptions = GroqTextGenOptions(
    model="llama3-70b-8192", temperature=0.3
)


class BaseGroqChat:
    """
    Base class for chatting using Groq Large Language Models (LLMs).

    :param llm: An instance of the async Groq client.
    :type llm: :class:`groq.AsyncGroq`
    :param llm_options: A dictionary of options to pass to the Groq LLM.
    :type llm_options: :class:`llmsmith.task.textgen.options.groq.GroqTextGenOptions`, optional
    """

    def __init__(
        self,
        llm: groq.AsyncGroq,
        llm_options: GroqTextGenOptions = default_options,
    ) -> None:
        self.llm: groq.AsyncGroq = llm
        self.llm_options: GroqTextGenOptions = llm_options or default_options

    async def chat(
        self,
        messages_payload: List[Message],
        tools: Union[List[Tool], None] = None,
    ) -> ChatResponse:
        """
        Generates text using Groq LLM using the given input.

        :param messages_payload: The input messages for the chat.
        :type messages_payload: List[:class:`groq.types.chat.completion_create_params.Message`]
        :param tools: Tools (functions) which can be used by the LLM.
        :type tools: List[:class:`groq.types.chat.completion_create_params.Tool`], optional
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
            f"Groq chat request: PAYLOAD: {messages_payload}\n OPTIONS: {chat_completion_options}"
        )

        llm_reply: ChatCompletion = await self.llm.chat.completions.create(
            messages=messages_payload, tools=tools, **chat_completion_options
        )

        log.debug(f"Groq chat response: {llm_reply}")

        output_choice_with_func_call = next(
            (c for c in llm_reply.choices if c.finish_reason == "tool_calls"),
            None,
        )

        if output_choice_with_func_call:
            return ChatResponse(
                text=output_choice_with_func_call.message.content,
                raw_output=llm_reply,
                function_calls={
                    tool.id: FunctionCall(
                        id=tool.id,
                        name=tool.function.name,
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


class GroqTextGenTask(Task[str, str]):
    """
    Task for generating text using Groq Cloud's Large Language Models (LLMs).

    :param name: The name of the task.
    :type name: str
    :param llm: An instance of the Async Groq client.
    :type llm: :class:`groq.AsyncGroq`
    :param llm_options: A dictionary of options to pass to the LLM in Groq Cloud.
    :type llm_options: :class:`llmsmith.task.textgen.options.groq.GroqTextGenOptions`, optional
    :raises ValueError: If the name is empty.
    """

    def __init__(
        self,
        name: str,
        llm: groq.AsyncGroq,
        llm_options: GroqTextGenOptions = default_options,
    ) -> None:
        super().__init__(name)
        self._chat = BaseGroqChat(llm, llm_options)

    async def execute(self, task_input: TaskInput[str]) -> TaskOutput[str]:
        """
        Generates text using Groq LLM using the given input.

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
