from typing import Any, TypedDict, Union

try:
    from google.generativeai.types.generation_types import GenerationConfigType
    from google.generativeai.types.safety_types import SafetySettingOptions
except ImportError:
    raise ImportError(
        "The 'google.generativeai' library is required to use Gemini LLMs. You can install it with `pip install \"llmsmith[gemini]\"`"
    )


class GeminiTextGenOptions(TypedDict):
    """
    A dictionary of options to pass to the Google Gemini LLM for text generation.
    The option names are same as the ones used in Gemini client.
    Refer below links for more info.

    * https://github.com/google/generative-ai-python/blob/v0.4.0/google/generativeai/types/generation_types.py
    * https://github.com/google/generative-ai-python/blob/v0.4.0/google/generativeai/types/safety_types.py
    * https://github.com/google/generative-ai-python/blob/v0.4.0/google/generativeai/types/content_types.py
    * https://ai.google.dev/tutorials/python_quickstart
    """

    generation_config: Union[GenerationConfigType, None]
    safety_settings: Union[SafetySettingOptions, None]
    request_options: Union[dict[str, Any], None]


def _completion_create_options_dict(options: GeminiTextGenOptions) -> dict:
    return {attr: options.get(attr) for attr in GeminiTextGenOptions.__annotations__}
