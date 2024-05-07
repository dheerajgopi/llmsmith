import logging
from typing import List, Union

from llmsmith.task.textgen.errors import TextGenFailedException
from llmsmith.task.textgen.options.cohere import CohereTextGenOptions

try:
    import cohere
    from cohere.types.non_streamed_chat_response import NonStreamedChatResponse
    from cohere.types.chat_message import ChatMessage
    from cohere.types.tool import Tool
    from cohere.types.chat_request_tool_results_item import ChatRequestToolResultsItem
except ImportError:
    raise ImportError(
        "The 'cohere' library is required to use Cohere LLMs. You can install it with `pip install \"llmsmith[cohere]\"`"
    )

from llmsmith.task.base import Task
from llmsmith.task.models import ChatResponse, FunctionCall, TaskInput, TaskOutput
from llmsmith.task.textgen.options.cohere import (
    _chat_options_dict,
)


log = logging.getLogger(__name__)

# Default options for text generation using Cohere's LLMs.
default_options: CohereTextGenOptions = CohereTextGenOptions(
    model="command-r-plus", temperature=0.3
)


class BaseCohereChat:
    """
    Base class for chatting using Cohere Large Language Models (LLMs).

    :param llm: An instance of the Async Cohere client.
    :type llm: :class:`cohere.AsyncClient`
    :param llm_options: A dictionary of options to pass to the Cohere LLM.
    :type llm_options: :class:`llmsmith.task.textgen.options.cohere.CohereTextGenOptions`, optional
    """

    def __init__(
        self,
        llm: cohere.AsyncClient,
        llm_options: CohereTextGenOptions = default_options,
    ) -> None:
        self.llm: cohere.AsyncClient = llm
        self.llm_options: CohereTextGenOptions = llm_options or default_options

    async def chat(
        self,
        message: str,
        chat_history: Union[List[ChatMessage], None] = None,
        conversation_id: str = None,
        tools: Union[List[Tool], None] = None,
        tool_results: Union[List[ChatRequestToolResultsItem], None] = None,
    ) -> ChatResponse:
        """
        Generates text using Cohere LLM using the given input.

        :param message: The input message for the chat.
        :type message: str
        :param chat_history: Chat history
        :type chat_history: List[ChatMessage], optional
        :param tools: Tools (functions) which can be used by the LLM.
        :type tools: List[:class:`cohere.types.tool.Tool`], optional
        :param tool_results: Output of the tools to be sent to the LLM.
        :type tool_results: List[:class:`cohere.types.chat_request_tool_results_item.ChatRequestToolResultsItem`], optional
        :raises TextGenFailedError: If AI fails to generate text based on the prompt.
        :returns: chat response from the LLM.
        :rtype: :class:`llmsmith.task.models.ChatResponse`
        """
        chat_options: dict = _chat_options_dict(self.llm_options)

        log.debug(f"Cohere chat request: PAYLOAD: {message}\n OPTIONS: {chat_options}")

        llm_reply: NonStreamedChatResponse = await self.llm.chat(
            message=message,
            chat_history=chat_history,
            conversation_id=conversation_id,
            tools=tools,
            tool_results=tool_results,
            **chat_options,
        )

        log.debug(f"Cohere chat response: {llm_reply}")

        if llm_reply.finish_reason != "COMPLETE":
            raise TextGenFailedException(
                "Failed to generate text",
                failure_reason=self._block_reason_str(llm_reply),
            )

        function_calls: dict[str, FunctionCall] = (
            {
                tool_call.name: FunctionCall(id=None, args=tool_call.parameters or {})
                for tool_call in llm_reply.tool_calls
            }
            if llm_reply.tool_calls
            else None
        )

        log.debug(f"chat response output value: {llm_reply.text}")

        return ChatResponse(
            text=llm_reply.text, raw_output=llm_reply, function_calls=function_calls
        )

    @classmethod
    def _block_reason_str(cls, llm_reply: NonStreamedChatResponse) -> str:
        if llm_reply.finish_reason == "ERROR_TOXIC":
            return "SAFETY_CHECK_FAILED"

        if llm_reply.finish_reason == "ERROR_LIMIT":
            return "LIMIT_EXCEEDED"

        if llm_reply.finish_reason == "MAX_TOKENS":
            return "MAX_TOKENS_REACHED"

        if llm_reply.finish_reason == "USER_CANCEL":
            return "CANCELLED"

        return "OTHER"


class CohereTextGenTask(Task[str, str]):
    """
    Task for generating text using Cohere's Large Language Models (LLMs).

    :param name: The name of the task.
    :type name: str
    :param llm: An instance of the Async Cohere client.
    :type llm: :class:`cohere.AsyncClient`
    :param llm_options: A dictionary of options to pass to the Cohere LLM.
    :type llm_options: :class:`llmsmith.task.textgen.options.cohere.CohereTextGenOptions`, optional
    :raises ValueError: If the name is empty.
    """

    def __init__(
        self,
        name: str,
        llm: cohere.AsyncClient,
        llm_options: CohereTextGenOptions = default_options,
    ) -> None:
        super().__init__(name)
        self._chat = BaseCohereChat(llm, llm_options)

    async def execute(self, task_input: TaskInput[str]) -> TaskOutput[str]:
        """
        Generates text using Cohere LLM using the given input.

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

        chat_response = await self._chat.chat(message=llm_input_content)

        return TaskOutput(
            content=chat_response.text, raw_output=chat_response.raw_output
        )
