import logging
from typing import List, Union
from llmsmith.llm.base import ChatLLM
from llmsmith.llm.models import LLMChatInput, LLMChatReply, LLMChatResponseContent

try:
    import anthropic
    from anthropic.types.message import Message
    from anthropic._types import NOT_GIVEN
except ImportError:
    raise ImportError(
        "The 'anthropic' library is required to use Claude. You can install it with `pip install llmsmith[claude]`"
    )


log = logging.getLogger(__name__)


class Claude(ChatLLM):
    """Anthropic Claude specific implementation of :class:`llmsmith.llm.base.ChatLLM`.

    :param api_key: Anthropic API key. If `None` is provided, it will fetch the value of `ANTHROPIC_API_KEY` environment variable.
    :type api_key: str, optional
    :param auth_token: Anthropic auth token. If `None` is provided, it will fetch the value of `ANTHROPIC_AUTH_TOKEN` environment variable.
    :type auth_token: str, optional
    :param base_url: Base URL for Claude LLM. If `None` is provided, it defaults to `https://api.anthropic.com`.
    :type base_url: str, optional
    :param default_model: Claude LLM model to be used by default. Defaults to `claude-instant-1.2`.
    :type default_model: str, optional
    :param default_temperature: Claude LLM model temperature to be set by default. Defaults to `0.3`.
    :type default_temperature: float, optional
    """

    def __init__(
        self,
        api_key: Union[str, None] = None,
        auth_token: Union[str, None] = None,
        base_url: Union[str, None] = None,
        default_model: str = "claude-instant-1.2",
        default_temperature: float = 0.3,
    ) -> None:
        self.default_model: str = default_model
        self.default_temperature: str = default_temperature
        self.client = anthropic.Anthropic(
            api_key=api_key, auth_token=auth_token, base_url=base_url
        )

    def chat(self, messages: LLMChatInput, **kwargs: dict[str, any]) -> LLMChatReply:
        """Send messages (prompt) to the Anthropic Claude LLM and return its response.

        :param messages: The messages to be sent to the LLM
        :type messages: :class:`llmsmith.llm.models.LLMChatInput`
        :keyword str model: Anthropic Claude LLM model to be used for the chat. Overrides the default model.
        :keyword float temperature: Anthropic Claude LLM model temperature to be set for the chat. Overrides the default temperature value.
        :keyword int max_tokens: Max number of tokens to generate before stopping.
        :keyword str system_prompt: Can be used to set system prompt. Optional. Will not set any system prompt by default.
        :return: Contains the text content replied by the AI, along with the actual response
            object (:class:`anthropic.types.message.Message`) returned by the Anthropic client.
        :rtype: :class:`llmsmith.llm.models.LLMChatReply`
        """
        model_name: str = kwargs.get("model", self.default_model)
        temp: float = kwargs.get("temperature", self.default_temperature)
        max_tokens: int = kwargs.get("max_tokens", 1024)
        sys_prompt: Union[str, None] = kwargs.get("system_prompt")

        messages_payload: List[dict] = [
            {"role": msg.role, "content": msg.content} for msg in messages.messages
        ]

        log.debug(
            f"Anthropic Claude chat request: PAYLOAD: {messages_payload}\n OPTIONS: {kwargs}"
        )

        completions: Message = self.client.messages.create(
            system=sys_prompt if sys_prompt else NOT_GIVEN,
            max_tokens=max_tokens,
            messages=messages_payload,
            model=model_name,
            temperature=temp,
        )

        log.debug(f"Anthropic Claude chat response: {completions}")

        return LLMChatReply(
            content=[
                LLMChatResponseContent(content=content.text, type="text")
                for content in completions.content
            ],
            internal_response=completions,
        )
