from typing import Dict, List, TypedDict, Union

try:
    from groq.types.chat.completion_create_params import ResponseFormat, ToolChoice
except ImportError:
    raise ImportError(
        "The 'groq' library is required to use LLMs in Groq. You can install it with `pip install \"llmsmith[groq]\"`"
    )


class GroqTextGenOptions(TypedDict):
    """
    A dictionary of options to pass to the Groq LLM for text generation.
    The option names are same as the ones used in Groq client (except `system_prompt`, which is an extra).
    Refer below links for more info.

    * https://github.com/groq/groq-python/blob/v0.6.0/src/groq/types/chat/completion_create_params.py
    * https://console.groq.com/docs/text-chat
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
    tool_choice: Union[ToolChoice, None]
    top_logprobs: Union[int, None]
    top_p: Union[float, None]
    user: Union[str, None]
    # Timeout to be set for Groq API calls.
    timeout: Union[float, None]


def _completion_create_options_dict(options: GroqTextGenOptions) -> dict:
    opt = {
        attr: options.get(attr)
        for attr in GroqTextGenOptions.__annotations__
        if attr not in ["system_prompt"]
    }

    if not opt.get("model"):
        opt["model"] = "llama3-70b-8192"

    if not opt.get("tool_choice"):
        opt["tool_choice"] = "auto"

    return opt
