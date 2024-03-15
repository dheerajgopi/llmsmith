from dataclasses import dataclass
from typing import List


@dataclass
class LLMChatResponseContent:
    """
    Contains the content and its type for each individual response choice in the LLM reply.
    In case of chat response, value of `type` will always be `text`.
    """

    content: str
    type: str


@dataclass
class LLMChatReply:
    """Messages replied by the LLM during chat."""

    content: List[LLMChatResponseContent]
    internal_response: any


@dataclass
class LLMChatMessage:
    """Contains the content and role for individual messages to be sent to the LLM for chatting."""

    content: str
    role: str


@dataclass
class LLMChatInput:
    """Messages to be passed as input for chatting with LLMs."""

    messages: List[LLMChatMessage]
