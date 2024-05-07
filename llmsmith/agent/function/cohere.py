try:
    import cohere
    from cohere import ChatRequestToolResultsItem
    from cohere.types.tool import Tool
    from cohere.types.tool_call import ToolCall
    from cohere.types.chat_message import ChatMessage
    from cohere.types.non_streamed_chat_response import NonStreamedChatResponse
except ImportError:
    raise ImportError(
        "The 'cohere' library is required to use Cohere LLMs. You can install it with `pip install \"llmsmith[cohere]\"`"
    )

import logging
from typing import Callable, List

from llmsmith.agent.errors import MaxTurnsReachedException
from llmsmith.agent.tool.cohere import CohereTool
from llmsmith.task.base import Task
from llmsmith.task.models import ChatResponse, TaskInput, TaskOutput
from llmsmith.task.textgen.cohere import BaseCohereChat
from llmsmith.task.textgen.options.cohere import CohereTextGenOptions


log = logging.getLogger(__name__)


class CohereFunctionAgent(Task[str, str]):
    """
    Agent based on function calling capability of Cohere LLMs.
    Agent will loop until one of the conditions are met:

    * If the LLM does not choose any tools (function) and simply returns text content in the response.
    * If maximum number of turns are reached in the agent loop.

    :param name: The name of the task.
    :type name: str
    :param llm: An instance of the Cohere client.
    :type llm: :class:`cohere.AsyncClient`
    :param llm_options: A dictionary of options to pass to the Cohere LLM.
    :type llm_options: :class:`llmsmith.task.textgen.options.cohere.CohereTextGenOptions`, optional
    :param tools: List of tools (functions) which can be used by the Cohere LLMs. Each tool contains both the declaration and the actual callable.
    :type tools: :class:`llmsmith.agent.tool.cohere.CohereTool`
    :param max_turns: Maximum number of turns allowed in the agent loop. Defaults to 5.
    :type max_turns: str
    :raises ValueError: If the name is empty or if max_turns is less than 1.
    """

    def __init__(
        self,
        name: str,
        llm: cohere.AsyncClient,
        llm_options: CohereTextGenOptions,
        tools: List[CohereTool] = [],
        max_turns: int = 5,
    ) -> None:
        super().__init__(name)

        if max_turns <= 0:
            raise ValueError("max_turns should be 1 or above")

        self._chat = BaseCohereChat(llm, llm_options)
        self.max_turns = max_turns

        # Convert to Tool class if declaration is of dict type
        for tool in tools:
            if isinstance(tool.declaration, dict):
                tool.declaration = Tool(
                    name=tool.declaration.get("name"),
                    description=tool.declaration.get("description"),
                    parameter_definitions=tool.declaration.get("parameter_definitions"),
                )

        self.llm_tools = [tool.declaration for tool in tools]
        self._tool_callables: dict[str, Callable] = {
            tool.declaration.name: tool.callable for tool in tools
        }

    async def execute(self, task_input: TaskInput[str]) -> TaskOutput[str]:
        """
        Executes the agent by running the loop. The agent exits the loop and returns the LLM response
        only if no more function calls are required.

        :param task_input: The input to the agent task.
        :type task_input: :class:`llmsmith.task.models.TaskInput[str]`
        :raises ValueError: If the content of the task input is not a string.
        :raises PromptBlockedError: If the prompt is blocked by the AI.
        :raises TextGenFailedError: If AI fails to generate text based on the prompt.
        :raises MaxTurnsReachedException: If maximum number of turns are reached in the agent loop.
        :returns: The output of the task.
        :rtype: :class:`llmsmith.task.models.TaskOutput[str]`
        """
        if not isinstance(task_input.content, str):
            log.error(f"task_input value: {task_input}")
            raise ValueError("task_input.content should be of type 'str'")

        llm_input_content: str = task_input.content
        func_results: List[ChatRequestToolResultsItem] = []
        chat_hist: List[ChatMessage] = []

        for turn in range(self.max_turns):
            chat_response: ChatResponse = await self._chat.chat(
                message=llm_input_content,
                tools=self.llm_tools,
                tool_results=func_results or None,
                chat_history=chat_hist or None,
            )

            raw_output: NonStreamedChatResponse = chat_response.raw_output
            chat_hist = raw_output.chat_history
            func_calls = chat_response.function_calls or {}

            # Exit condition: If no function calls are required.
            if not func_calls:
                log.debug(
                    f"CohereFunctionAgent turn-{turn+1} | Exiting agent loop with text output: f{chat_response.text}"
                )
                return TaskOutput(
                    content=chat_response.text,
                    raw_output=chat_response.raw_output,
                )

            log.debug(
                f"CohereFunctionAgent turn-{turn+1} | Function call required: {func_calls}"
            )

            # Execute functions (tools)
            for func_name, func in func_calls.items():
                func_output = self._tool_callables[func_name](**func.args)
                tool_call = ToolCall(name=func_name, parameters=func.args)
                func_results.append(
                    {"call": tool_call, "outputs": [{"answer": func_output}]}
                )

        raise MaxTurnsReachedException()
