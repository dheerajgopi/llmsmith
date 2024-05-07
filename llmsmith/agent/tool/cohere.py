from typing import Callable


try:
    from cohere.types.tool import Tool
except ImportError:
    raise ImportError(
        "The 'cohere' library is required to use Cohere LLMs. You can install it with `pip install \"llmsmith[cohere]\"`"
    )


class CohereTool:
    """
    Wrapper for both function declaration and the actual callable to be used in Cohere chat APIs.

    :param declaration: Function declaration to be passed into Cohere client.
    :type declaration: :class:`cohere.types.tool.Tool`
    :param callable: Actual function (callable)
    :type callable: :class:`typing.Callable`
    """

    def __init__(self, declaration: Tool, callable: Callable) -> None:
        if not declaration or not callable:
            raise ValueError("Both 'declaration' and 'callable' params are mandatory")

        self.declaration: Tool = declaration
        self.callable: Callable = callable
