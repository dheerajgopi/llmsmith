from typing import List, Mapping, TypedDict, Union

try:
    from anthropic.types.message_create_params import Metadata
except ImportError:
    raise ImportError(
        "The 'anthropic' library is required to use Claude LLMs. You can install it with `pip install \"llmsmith[claude]\"`"
    )


class ClaudeTextGenOptions(TypedDict):
    """
    A dictionary of options to pass to the Anthropic Claude LLM for text generation.
    The option names are same as the ones used in Anthropic client.
    Refer below links for more info.

    * https://github.com/anthropics/anthropic-sdk-python/blob/v0.19.2/src/anthropic/types/message_create_params.py
    * https://docs.anthropic.com/claude/reference/messages_post
    """

    model: str
    system: Union[str, None]
    max_tokens: Union[int, None]
    metadata: Union[Metadata, None]
    stop_sequences: Union[List[str], None]
    temperature: Union[float, None]
    top_k: Union[int, None]
    top_p: Union[float, None]
    extra_headers: Mapping[str, Union[str, None]]
    extra_query: Mapping[str, object]
    extra_body: object
    timeout: Union[float, None]


def _completion_create_options_dict(options: ClaudeTextGenOptions) -> dict:
    opt = {
        attr: options.get(attr)
        for attr in ClaudeTextGenOptions.__annotations__
        if options.get(attr)
    }
    if not opt.get("model"):
        opt["model"] = "claude-3-opus-20240229"
    if not opt.get("max_tokens"):
        opt["max_tokens"] = 1024

    return opt
