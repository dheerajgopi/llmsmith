class TextGenException(Exception):
    """Base exception for all text generation related errors"""


class TextGenFailedException(TextGenException):
    """Raised when the AI fails to generate text for some reason (like unsafe prompt, exceeded token count etc.)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.failure_reason = kwargs.get("failure_reason")


class PromptBlockedException(TextGenException):
    """Raised when the prompt is blocked by the AI for some reason (like blocked prompt for example)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.block_reason = kwargs.get("block_reason")
