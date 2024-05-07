from typing import Callable


try:
    from google.ai.generativelanguage_v1beta.types import FunctionDeclaration
except ImportError:
    raise ImportError(
        "The 'google.generativeai' library is required to use Gemini LLMs. You can install it with `pip install \"llmsmith[gemini]\"`"
    )


class GeminiTool:
    """
    Wrapper for both function declaration and the actual callable to be used in Gemini LLMs.

    :param declaration: Function declaration to be passed into Gemini client.
    :type declaration: :class:`google.ai.generativelanguage_v1beta.types.FunctionDeclaration`
    :param callable: Actual function (callable)
    :type callable: :class:`typing.Callable`
    """

    def __init__(self, declaration: FunctionDeclaration, callable: Callable) -> None:
        if not declaration or not callable:
            raise ValueError("Both 'declaration' and 'callable' params are mandatory")

        self.declaration: FunctionDeclaration = declaration
        self.callable: Callable = callable
