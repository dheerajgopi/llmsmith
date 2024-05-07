from typing import Dict, List, TypedDict, Union

try:
    from openai.types.chat.completion_create_params import ResponseFormat
    from openai.types.chat.chat_completion_tool_choice_option_param import (
        ChatCompletionToolChoiceOptionParam,
    )
except ImportError:
    raise ImportError(
        "The 'openai' library is required to use OpenAI LLMs. You can install it with `pip install \"llmsmith[openai]\"`"
    )


class OpenAITextGenOptions(TypedDict):
    """
    A dictionary of options to pass to the OpenAI LLM for text generation.
    The option names are same as the ones used in OpenAI client (except `system_prompt`, which is an extra).
    Refer below links for more info.

    * https://github.com/openai/openai-python/blob/v1.13.3/src/openai/types/chat/completion_create_params.py
    * https://platform.openai.com/docs/api-reference/chat/create
    """

    model: str
    # System prompt to be set in the chat creation request.
    system_prompt: Union[str, None]
    frequency_penalty: Union[float, None]
    logit_bias: Union[Dict[str, int], None]
    logprobs: Union[bool, None]
    max_tokens: Union[int, None]
    presence_penalty: Union[float, None]
    response_format: Union[ResponseFormat, None]
    seed: Union[int, None]
    stop: Union[str, List[str], None]
    temperature: Union[float, None]
    tool_choice: Union[ChatCompletionToolChoiceOptionParam, None]
    top_logprobs: Union[int, None]
    top_p: Union[float, None]
    user: Union[str, None]
    # Timeout to be set for OpenAI API calls.
    timeout: Union[float, None]


def _completion_create_options_dict(options: OpenAITextGenOptions) -> dict:
    opt = {
        attr: options.get(attr)
        for attr in OpenAITextGenOptions.__annotations__
        if attr not in ["system_prompt"]
    }

    if not opt.get("model"):
        opt["model"] = "gpt-3.5-turbo"

    return opt
