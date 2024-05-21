from typing import Callable


try:
    from groq.types.chat.completion_create_params import Tool
except ImportError:
    raise ImportError(
        "The 'groq' library is required to use LLMs in Groq. You can install it with `pip install \"llmsmith[groq]\"`"
    )


class GroqTool:
    """
    Wrapper for both function declaration and the actual callable to be used in chat completion APIs.

    :param declaration: Function declaration to be passed into Groq client.
    :type declaration: :class:`groq.types.chat.completion_create_params.Tool`
    :param callable: Actual function (callable)
    :type callable: :class:`typing.Callable`
    """

    def __init__(self, declaration: Tool, callable: Callable) -> None:
        if not declaration or not callable:
            raise ValueError("Both 'declaration' and 'callable' params are mandatory")

        self.declaration: Tool = declaration
        self.callable: Callable = callable
