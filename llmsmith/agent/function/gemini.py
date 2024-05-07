try:
    from google.ai.generativelanguage_v1beta.types import (
        Part,
        FunctionCall,
        FunctionDeclaration,
        FunctionResponse,
        Tool,
    )
    from google.generativeai import GenerativeModel
except ImportError:
    raise ImportError(
        "The 'google.generativeai' library is required to use Gemini LLMs. You can install it with `pip install \"llmsmith[gemini]\"`"
    )

import logging
from typing import Callable, List
from llmsmith.agent.errors import MaxTurnsReachedException
from llmsmith.agent.tool.gemini import GeminiTool
from llmsmith.task.base import Task
from llmsmith.task.models import TaskInput, TaskOutput
from llmsmith.task.textgen.gemini import BaseGeminiChat
from llmsmith.task.textgen.options.gemini import GeminiTextGenOptions


log = logging.getLogger(__name__)


class GeminiFunctionAgent(Task[str, str]):
    """
    Agent based on function calling capability of Gemini LLMs.
    Agent will loop until one of the conditions are met:

    * If the LLM does not choose any tools (function) and simply returns text content in the response.
    * If maximum number of turns are reached in the agent loop.

    :param name: The name of the task.
    :type name: str
    :param llm: An instance of the Gemini client.
    :type llm: :class:`google.generativeai.GenerativeModel`
    :param llm_options: A dictionary of options to pass to the Gemini LLM.
    :type llm_options: :class:`llmsmith.task.textgen.options.gemini.GeminiTextGenOptions`, optional
    :param tools: List of tools (functions) which can be used by the Gemini LLMs. Each tool contains both the declaration and the actual callable.
    :type tools: :class:`llmsmith.agent.tool.gemini.GeminiTool`
    :param max_turns: Maximum number of turns allowed in the agent loop. Defaults to 5.
    :type max_turns: str
    :raises ValueError: If the name is empty or if max_turns is less than 1.
    """

    def __init__(
        self,
        name: str,
        llm: GenerativeModel,
        llm_options: GeminiTextGenOptions,
        tools: List[GeminiTool] = [],
        max_turns: int = 5,
    ) -> None:
        super().__init__(name)

        if max_turns <= 0:
            raise ValueError("max_turns should be 1 or above")

        self._chat = BaseGeminiChat(llm, llm_options)
        self.max_turns = max_turns

        # Convert to FunctionDeclaration if declaration is of dict type
        for tool in tools:
            if isinstance(tool.declaration, dict):
                tool.declaration = FunctionDeclaration(tool.declaration)

        function_declarations = [tool.declaration for tool in tools]
        self.llm_tools = (
            Tool(function_declarations=function_declarations)
            if function_declarations
            else None
        )
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
        messages_payload: List[dict] = [{"role": "user", "parts": [llm_input_content]}]

        for turn in range(self.max_turns):
            chat_response = await self._chat.chat(
                messages_payload=messages_payload, tools=self.llm_tools
            )

            func_calls = chat_response.function_calls or {}

            # Exit condition: If no function calls are required.
            if not func_calls:
                log.debug(
                    f"GeminiFunctionAgent turn-{turn+1} | Exiting agent loop with text output: f{chat_response.text}"
                )
                return TaskOutput(
                    content=chat_response.text,
                    raw_output=chat_response.raw_output,
                )

            log.debug(
                f"GeminiFunctionAgent turn-{turn+1} | Function call required: {func_calls}"
            )

            for func_name, func in func_calls.items():
                func_output = self._tool_callables[func_name](**func.args)
                messages_payload.append(
                    {
                        "role": "function",
                        "parts": [
                            Part(
                                function_call=FunctionCall(
                                    name=func_name, args=func.args
                                )
                            )
                        ],
                    }
                )
                messages_payload.append(
                    {
                        "role": "function",
                        "parts": [
                            Part(
                                function_response=FunctionResponse(
                                    name=func_name, response={"result": func_output}
                                )
                            )
                        ],
                    }
                )

        raise MaxTurnsReachedException()
