import logging
from typing import List, Union
from llmsmith.llm.base import ChatLLM
from llmsmith.llm.models import LLMChatInput, LLMChatReply, LLMChatResponseContent

try:
    import openai
    from openai.types.chat.chat_completion import ChatCompletion
except ImportError:
    raise ImportError(
        "The 'openai' library is required to use OpenAI. You can install it with `pip install llmsmith[openai]`"
    )


log = logging.getLogger(__name__)


class OpenAI(ChatLLM):
    """OpenAI specific implementation of :class:`llmsmith.llm.base.ChatLLM`.

    :param api_key: OpenAI API key. If `None` is provided, it will fetch the value of `OPENAI_API_KEY` environment variable.
    :type api_key: str, optional
    :param base_url: Base URL for OpenAI LLM model. If `None` is provided, it defaults to `https://api.openai.com/v1`.
    :type base_url: str, optional
    :param default_model: OpenAI LLM model to be used by default. Defaults to `gpt-3.5-turbo`.
    :type default_model: str, optional
    :param default_temperature: OpenAI LLM model temperature to be set by default. Defaults to `0.7`.
    :type default_temperature: float, optional
    """

    def __init__(
        self,
        api_key: Union[str, None] = None,
        base_url: Union[str, None] = None,
        default_model: str = "gpt-3.5-turbo",
        default_temperature: float = 0.7,
    ) -> None:
        self.default_model: str = default_model
        self.default_temperature: str = default_temperature
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: LLMChatInput, **kwargs: dict[str, any]) -> LLMChatReply:
        """Send messages (prompt) to the OpenAI LLM and return its response.

        :param messages: The messages to be sent to the LLM
        :type messages: :class:`llmsmith.llm.models.LLMChatInput`
        :keyword str model: OpenAI LLM model to be used for the chat. Overrides the default model.
        :keyword float temperature: OpenAI LLM model temperature to be set for the chat. Overrides the default temperature value.
        :keyword str system_prompt: Can be used to set system prompt. Optional. Will not set any system prompt by default.
        :return: Contains the text content replied by the AI, along with the actual response
            object (:class:`openai.types.chat.chat_completion.ChatCompletion`) returned by the OpenAI client.
        :rtype: :class:`llmsmith.llm.models.LLMChatReply`
        """
        model_name: str = kwargs.get("model", self.default_model)
        temp: float = kwargs.get("temperature", self.default_temperature)
        sys_prompt: Union[str, None] = kwargs.get("system_prompt")
        messages_payload: List[dict] = []

        if sys_prompt:
            messages_payload.append({"role": "system", "content": sys_prompt})
        messages_payload.extend(
            [{"role": msg.role, "content": msg.content} for msg in messages.messages]
        )

        log.debug(
            f"OpenAI chat request: PAYLOAD: {messages_payload}\n OPTIONS: {kwargs}"
        )

        completions: ChatCompletion = self.client.chat.completions.create(
            model=model_name, messages=messages_payload, temperature=temp
        )

        log.debug(f"OpenAI chat response: {completions}")

        return LLMChatReply(
            content=[
                LLMChatResponseContent(content=choice.message.content, type="text")
                for choice in completions.choices
            ],
            internal_response=completions,
        )
