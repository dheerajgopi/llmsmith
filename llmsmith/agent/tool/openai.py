from typing import Callable, Union


try:
    from openai.types.shared_params import FunctionDefinition
    from openai.types.beta.assistant_tool_param import AssistantToolParam
except ImportError:
    raise ImportError(
        "The 'openai' library is required to use OpenAI LLMs. You can install it with `pip install \"llmsmith[openai]\"`"
    )


class OpenAIChatTool:
    """
    Wrapper for both function declaration and the actual callable to be used in chat completion APIs.

    :param declaration: Function declaration to be passed into OpenAI client.
    :type declaration: :class:`openai.types.shared_params.FunctionDefinition`
    :param callable: Actual function (callable)
    :type callable: :class:`typing.Callable`
    """

    def __init__(self, declaration: FunctionDefinition, callable: Callable) -> None:
        if not declaration or not callable:
            raise ValueError("Both 'declaration' and 'callable' params are mandatory")

        self.declaration: FunctionDefinition = declaration
        self.callable: Callable = callable


class OpenAIAssistantTool:
    """
    Wrapper for both tool declaration and the actual callable (required for function tools) to be used in assistant APIs.

    :param declaration: Function declaration to be passed into OpenAI client.
    :type declaration: :class:`openai.types.shared_params.FunctionDefinition`
    :param callable: Actual function (callable) if tool is of `function` type
    :type callable: :class:`typing.Callable`, optional
    """

    def __init__(self, declaration: AssistantToolParam, callable: Callable) -> None:
        if not declaration:
            raise ValueError("'declaration' param is mandatory")

        if declaration.get("type") == "function" and not callable:
            raise ValueError(
                "'callable' param is mandatory for tools of `function` type"
            )

        self.declaration: AssistantToolParam = declaration
        self.callable: Union[Callable, None] = callable
