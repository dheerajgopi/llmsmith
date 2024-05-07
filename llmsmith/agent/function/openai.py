try:
    import openai
    from openai.types.beta import Assistant, Thread
    from openai.types.beta.threads import Run
except ImportError:
    raise ImportError(
        "The 'openai' library is required to use OpenAI LLMs. You can install it with `pip install \"llmsmith[openai]\"`"
    )

import json
import logging
from typing import Callable, List

from llmsmith.agent.errors import MaxTurnsReachedException
from llmsmith.agent.function.options.openai import (
    OpenAIAssistantOptions,
    _create_assistant_options_dict,
)
from llmsmith.agent.tool.openai import OpenAIAssistantTool
from llmsmith.task.base import Task
from llmsmith.task.models import TaskInput, TaskOutput
from llmsmith.task.textgen.errors import TextGenFailedException


log = logging.getLogger(__name__)


class OpenAIFunctionAgent(Task[str, str]):
    """
    Agent based on function calling capability of OpenAI LLMs.
    Agent will loop until one of the conditions are met:

    * If the LLM does not choose any tools and simply returns text content in the response.
    * If maximum number of turns are reached in the agent loop.

    :param name: The name of the task.
    :type name: str
    :param llm: An instance of the async OpenAI client.
    :type llm: :class:`openai.AsyncOpenAI`
    :param assistant: The OpenAI assistant.
    :type assistant: :class:`openai.types.beta.Assistant`
    :param tools: List of tools (file search, code interpreter, functions) which can be used by the OpenAI LLMs. Tools of `function` type should have the actual callable too.
    :type tools: List[:class:`llmsmith.agent.tool.openai.OpenAIAssistantTool`]
    :param max_turns: Maximum number of turns allowed in the agent loop. Defaults to 5.
    :type max_turns: str
    :raises ValueError: If the name is empty or if max_turns is less than 1.
    """

    def __init__(
        self,
        name: str,
        llm: openai.AsyncOpenAI,
        assistant: Assistant,
        tools: List[OpenAIAssistantTool] = [],
        max_turns: int = 5,
    ) -> None:
        super().__init__(name)

        if max_turns <= 0:
            raise ValueError("max_turns should be 1 or above")

        self.llm = llm
        self._llm_tools = tools
        self._assistant = assistant
        self.max_turns = max_turns

        self._tool_callables: dict[str, Callable] = {
            tool.declaration["function"]["name"]: tool.callable
            for tool in tools
            if tool.declaration["type"] == "function"
        }

    @classmethod
    async def create(
        cls,
        name: str,
        llm: openai.AsyncOpenAI,
        assistant_options: OpenAIAssistantOptions,
        tools: List[OpenAIAssistantTool] = [],
        max_turns: int = 5,
    ):
        """
        Factory method for creating an instance of `OpenAIFunctionAgent`.

        :param name: The name of the task.
        :type name: str
        :param llm: An instance of the async OpenAI client.
        :type llm: :class:`openai.AsyncOpenAI`
        :param assistant_options: A dictionary of options to pass to the OpenAI assistant.
        :type assistant_options: :class:`llmsmith.agent.function.options.openai.OpenAIAssistantOptions`, optional
        :param tools: List of tools (file search, code interpreter, functions) which can be used by the OpenAI LLMs. Tools of `function` type should have the actual callable too.
        :type tools: List[:class:`llmsmith.agent.tool.openai.OpenAIAssistantTool`]
        :param max_turns: Maximum number of turns allowed in the agent loop. Defaults to 5.
        :type max_turns: str
        :raises ValueError: If the name is empty or if max_turns is less than 1.
        """

        llm_tools = [tool.declaration for tool in tools] if tools else None
        assistant_opts_with_tools = _create_assistant_options_dict(assistant_options)
        assistant_opts_with_tools["tools"] = llm_tools

        openai_assistant = await llm.beta.assistants.create(**assistant_opts_with_tools)

        agent = cls(
            name=name,
            llm=llm,
            assistant=openai_assistant,
            tools=tools,
            max_turns=max_turns,
        )

        return agent

    async def execute(self, task_input: TaskInput[str]) -> TaskOutput[str]:
        """
        Executes the agent by running the loop. The agent exits the loop and returns the LLM response
        only if no more function calls are required.

        :param task_input: The input to the agent task.
        :type task_input: :class:`llmsmith.task.models.TaskInput[str]`
        :raises ValueError: If the content of the task input is not a string.
        :raises TextGenFailedError: If AI fails to generate text based on the prompt.
        :raises MaxTurnsReachedException: If maximum number of turns are reached in the agent loop.
        :returns: The output of the task.
        :rtype: :class:`llmsmith.task.models.TaskOutput[str]`
        """
        if not isinstance(task_input.content, str):
            log.error(f"task_input value: {task_input}")
            raise ValueError("task_input.content should be of type 'str'")

        llm_input_content: str = task_input.content

        thread: Thread = await self.llm.beta.threads.create()
        await self.llm.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=llm_input_content
        )

        run: Run = await self.llm.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=self._assistant.id,
        )

        for turn in range(self.max_turns):
            messages = await self.llm.beta.threads.messages.list(
                thread_id=thread.id,
                run_id=run.id,
                order="desc",
                limit=1,
            )

            # exit agent with the LLM response if no function calls are required
            if run.status == "completed":
                assistant_res: str = next(
                    (
                        content.text.value
                        for content in messages.data[0].content
                        if content.type == "text"
                    ),
                    None,
                )

                log.debug(
                    f"OpenAIFunctionAgent turn-{turn+1} | Exiting agent loop with text output: f{assistant_res}"
                )

                return TaskOutput(
                    content=assistant_res,
                    raw_output=run,
                )

            # requires function calls. Call the functions and send their outputs to the LLM.
            if run.status == "requires_action":
                func_tool_outputs = []

                log.debug(
                    f"OpenAIFunctionAgent turn-{turn+1} | Function call required: {run.required_action.submit_tool_outputs.tool_calls}"
                )

                for tool in run.required_action.submit_tool_outputs.tool_calls:
                    args = (
                        json.loads(tool.function.arguments)
                        if tool.function.arguments
                        else {}
                    )
                    func_output = self._tool_callables[tool.function.name](**args)

                    func_tool_outputs.append(
                        {"tool_call_id": tool.id, "output": str(func_output)}
                    )

                run = await self.llm.beta.threads.runs.submit_tool_outputs_and_poll(
                    thread_id=thread.id, run_id=run.id, tool_outputs=func_tool_outputs
                )

                continue

            # Handle errors
            else:
                log.debug(
                    f"OpenAIFunctionAgent turn-{turn+1} | Exiting agent loop due to error | run status: f{run.status}"
                )

                if run.status == "failed":
                    log.error(
                        f"OpenAIFunctionAgent turn-{turn+1} | Run failure reason: f{run.last_error}"
                    )

                raise TextGenFailedException(
                    "Failed to generate text",
                    failure_reason=self.__failure_reason(run.status),
                )

        raise MaxTurnsReachedException()

    def __failure_reason(self, run: Run) -> str:
        if run.status == "cancelled":
            return "OPENAI__RUN_CANCELLED"

        if run.status == "failed":
            if not run.last_error:
                return "OPENAI__RUN_FAILED"
            if run.last_error.code == "invalid_prompt":
                return "INVALID_PROMPT"
            if run.last_error.code == "rate_limit_exceeded":
                return "RATE_LIMIT_EXCEEDED"
            if run.last_error.code == "server_error":
                return "SERVER_ERROR"

            return "OPENAI__RUN_FAILED"

        if run.status == "expired":
            return "OPENAI__RUN_EXPIRED"

        return "OPENAI__UNKNOWN_ERR"
