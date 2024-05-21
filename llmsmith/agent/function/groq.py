try:
    import groq
except ImportError:
    raise ImportError(
        "The 'groq' library is required to use LLMs in Groq. You can install it with `pip install \"llmsmith[groq]\"`"
    )

import json
import logging
from typing import Callable, List

from llmsmith.agent.errors import MaxTurnsReachedException
from llmsmith.agent.tool.groq import GroqTool
from llmsmith.task.base import Task
from llmsmith.task.models import ChatResponse, TaskInput, TaskOutput
from llmsmith.task.textgen.groq import BaseGroqChat
from llmsmith.task.textgen.options.groq import GroqTextGenOptions


log = logging.getLogger(__name__)


class GroqFunctionAgent(Task[str, str]):
    """
    Agent based on function calling capability of LLMs in Groq.
    Agent will loop until one of the conditions are met:

    * If the LLM does not choose any tools (function) and simply returns text content in the response.
    * If maximum number of turns are reached in the agent loop.

    :param name: The name of the task.
    :type name: str
    :param llm: An instance of the async Groq client.
    :type llm: :class:`groq.AsyncGroq`
    :param llm_options: A dictionary of options to pass to the LLM in Groq.
    :type llm_options: :class:`llmsmith.task.textgen.options.groq.GroqTextGenOptions`, optional
    :param tools: List of tools (functions) which can be used by the LLMs in Groq. Each tool contains both the declaration and the actual callable.
    :type tools: :class:`llmsmith.agent.tool.groq.GroqTool`
    :param max_turns: Maximum number of turns allowed in the agent loop. Defaults to 5.
    :type max_turns: str
    :raises ValueError: If the name is empty or if max_turns is less than 1.
    """

    def __init__(
        self,
        name: str,
        llm: groq.AsyncGroq,
        llm_options: GroqTextGenOptions,
        tools: List[GroqTool] = [],
        max_turns: int = 5,
    ) -> None:
        super().__init__(name)

        if max_turns <= 0:
            raise ValueError("max_turns should be 1 or above")

        self._chat = BaseGroqChat(llm, llm_options)
        self.max_turns = max_turns
        self.llm_tools = [tool.declaration for tool in tools] if tools else None
        self._tool_callables: dict[str, Callable] = {
            tool.declaration["function"]["name"]: tool.callable
            for tool in tools
            if tool.declaration["type"] == "function"
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
        messages_payload: List[dict] = [{"role": "user", "content": llm_input_content}]

        for turn in range(self.max_turns):
            chat_response: ChatResponse = await self._chat.chat(
                messages_payload=messages_payload,
                tools=self.llm_tools,
            )

            func_calls = chat_response.function_calls or {}

            # Exit condition: If no function calls are required.
            if not func_calls:
                log.debug(
                    f"GroqFunctionAgent turn-{turn+1} | Exiting agent loop with text output: {chat_response.text}"
                )
                return TaskOutput(
                    content=chat_response.text,
                    raw_output=chat_response.raw_output,
                )

            log.debug(
                f"GroqFunctionAgent turn-{turn+1} | Function call required: {func_calls}"
            )

            # Add required function calls in the messages payload required for the next LLM call
            messages_payload.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "function": {
                                "name": func.name,
                                "arguments": json.dumps(func.args),
                            },
                            "type": "function",
                        }
                        for tool_id, func in func_calls.items()
                    ],
                }
            )

            # Execute functions (tools)
            for tool_id, func in func_calls.items():
                func_output = self._tool_callables[func.name](**func.args)
                messages_payload.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": func.name,
                        "content": str(func_output),
                    }
                )

        raise MaxTurnsReachedException()
