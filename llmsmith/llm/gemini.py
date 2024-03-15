import logging
from typing import List, Union
from llmsmith.llm.base import ChatLLM
from llmsmith.llm.models import LLMChatInput, LLMChatReply, LLMChatResponseContent

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "The 'google.generativeai' library is required to use Gemini. You can install it with `pip install llmsmith[gemini]`"
    )


log = logging.getLogger(__name__)


class Gemini(ChatLLM):
    """Google Gemini specific implementation of :class:`llmsmith.llm.base.ChatLLM`.

    :param api_key: Gemini API key. If `None` is provided, it will fetch the value of `GOOGLE_API_KEY` environment variable.
    :type api_key: str, optional
    :param model: Gemini LLM model to be used. Defaults to `gemini-1.0-pro`.
    :type model: str, optional
    :param default_generation_config: Gemini specific generation config to be set by default. Defaults to None.
        Please refer `google.generativeai.types.GenerationConfigType` for more details.
    :type default_generation_config: :class:`google.generativeai.types.GenerationConfigType`, optional
    :param default_safety_settings: Gemini specific safety settings to be set by default. Defaults to None.
        Please refer `google.generativeai.types.SafetySettingDict` for more details.
    :type default_safety_settings: :class:`google.generativeai.types.SafetySettingDict`, optional
    :param default_tools: Tools which can be used by Gemini during generation by default. Defaults to None.
        Please refer :class:`google.generativeai.types.FunctionLibraryType` for more details.
    :type default_tools: :class:`google.generativeai.types.FunctionLibraryType`, optional
    """

    def __init__(
        self,
        api_key: Union[str, None] = None,
        model: str = "gemini-1.0-pro",
        default_generation_config: Union[genai.types.GenerationConfigType, None] = None,
        default_safety_settings: Union[genai.types.SafetySettingDict, None] = None,
        default_tools: Union[genai.types.FunctionLibraryType, None] = None,
    ) -> None:
        self.model: str = model
        genai.configure(api_key=api_key)
        self.gen_model = genai.GenerativeModel(
            model_name=model,
            safety_settings=default_safety_settings,
            generation_config=default_generation_config,
            tools=default_tools,
        )

    def chat(self, messages: LLMChatInput, **kwargs: dict[str, any]) -> LLMChatReply:
        """Send messages (prompt) to the OpenAI LLM and return its response.

        :param messages: The messages to be sent to the LLM.
        :type messages: :class:`llmsmith.llm.models.LLMChatInput`
        :keyword google.generativeai.types.GenerationConfigType generation_config: Gemini specific generation config.
            Overrides the default generation config.
        :keyword google.generativeai.types.SafetySettingDict safety_settings: Gemini specific safety settings.
            Overrides the default safety settings.
        :keyword google.generativeai.types.FunctionLibraryType tools: Tools which can be used by Gemini.
            Overrides the default set of tools.
        :return: Contains the text content replied by the AI, along with the actual response
            object (:class:`google.generativeai.types.GenerateContentResponse`) returned by the Google Generative AI client.
        :rtype: :class:`llmsmith.llm.models.LLMChatReply`
        """
        generation_config: Union[genai.types.GenerationConfigType, None] = kwargs.get(
            "generation_config"
        )
        safety_settings: Union[genai.types.SafetySettingDict, None] = kwargs.get(
            "safety_settings"
        )
        tools: Union[genai.types.FunctionLibraryType, None] = kwargs.get("tools")

        messages_payload: List[dict] = [
            {"role": msg.role, "parts": [msg.content]} for msg in messages.messages
        ]

        log.debug(
            f"Google Gemini chat request: PAYLOAD: {messages_payload}\n OPTIONS: {kwargs}"
        )

        completions: genai.types.GenerateContentResponse = (
            self.gen_model.generate_content(
                contents=messages_payload,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools,
            )
        )

        log.debug(f"Google Gemini chat response: {completions}")

        return LLMChatReply(
            content=[
                LLMChatResponseContent(content=candidate.content, type="text")
                for candidate in completions.candidates
            ],
            internal_response=completions,
        )
