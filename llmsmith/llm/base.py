from abc import ABC, abstractmethod

from llmsmith.llm.models import LLMChatInput, LLMChatReply


class ChatLLM(ABC):
    """This is the abstract base class for all chat based LLM clients"""

    @abstractmethod
    def chat(self, messages: LLMChatInput, **kwargs: dict[str, any]) -> LLMChatReply:
        """Send messages (prompt) to the LLM and return the models response.

        :param messages: The messages to be sent to the LLM
        :type messages: :class:`llmsmith.llm.models.LLMChatInput`
        :param kwargs: Use this to pass LLM implementation specific parameters
        :type kwargs: dict[str, any]
        :return: Contains the text content replied by the AI, along with the actual response object returned by the underlying LLM client.
        :rtype: :class:`llmsmith.llm.models.LLMChatReply`
        """
        pass
