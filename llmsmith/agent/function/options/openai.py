from typing import TypedDict, Union

try:
    from openai._types import Headers, Query, Body
    from openai.types.beta.assistant_create_params import ToolResources
    from openai.types.beta.assistant_response_format_option_param import (
        AssistantResponseFormatOptionParam,
    )
except ImportError:
    raise ImportError(
        "The 'openai' library is required to use OpenAI LLMs. You can install it with `pip install \"llmsmith[openai]\"`"
    )


class OpenAIAssistantOptions(TypedDict):
    """
    A dictionary of options to be passed into the OpenAI assistant APIs.
    The option names are same as the ones used in OpenAI client (except `system_prompt`, which replaces `instructions` option).
    Refer below links for more info.

    * https://github.com/openai/openai-python/blob/v1.23.2/src/openai/types/beta/assistant_create_params.py
    * https://platform.openai.com/docs/assistants/overview?context=without-streaming
    """

    model: str
    description: Union[str, None]
    # System prompt to be set in the chat creation request.
    system_prompt: Union[str, None]
    metadata: Union[object, None]
    name: Union[str, None]
    response_format: Union[AssistantResponseFormatOptionParam, None]
    temperature: Union[float, None]
    tool_resources: Union[ToolResources, None]
    top_p: Union[float, None]
    extra_headers: Union[Headers, None]
    extra_query: Union[Query, None]
    extra_body: Union[Body, None]
    # Timeout to be set for OpenAI API calls.
    timeout: Union[float, None]


def _create_assistant_options_dict(options: OpenAIAssistantOptions = {}) -> dict:
    if not options:
        options = {}

    opt = {
        attr: options.get(attr)
        for attr in OpenAIAssistantOptions.__annotations__
        if attr not in ["system_prompt"]
    }

    if options.get("system_prompt"):
        opt["instructions"] = options.get("system_prompt")

    if not opt.get("model"):
        opt["model"] = "gpt-4-turbo"

    return opt
