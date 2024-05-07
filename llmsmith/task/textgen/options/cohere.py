from typing import List, TypedDict, Union

try:
    from cohere.base_client import OMIT
    from cohere.types.chat_request_prompt_truncation import ChatRequestPromptTruncation
    from cohere.types.chat_connector import ChatConnector
    from cohere.types.chat_document import ChatDocument
    from cohere.core.request_options import RequestOptions
except ImportError:
    raise ImportError(
        "The 'cohere' library is required to use Cohere LLMs. You can install it with `pip install \"llmsmith[cohere]\"`"
    )


class CohereTextGenOptions(TypedDict):
    """
    A dictionary of options to pass to the Cohere client for text generation.
    The option names are same as the ones used in Cohere client (except `system_prompt`, which replaces `preamble`).
    Refer below link for more info.

    * https://docs.cohere.com/reference/chat
    """

    model: str
    # System prompt to be set in the chat creation request.
    system_prompt: Union[str, None]
    model: Union[str, None]
    conversation_id: Union[str, None]
    prompt_truncation: Union[ChatRequestPromptTruncation, None]
    connectors: Union[List[ChatConnector], None]
    search_queries_only: Union[bool, None]
    documents: Union[List[ChatDocument], None]
    temperature: Union[float, None]
    max_tokens: Union[int, None]
    max_input_tokens: Union[int, None]
    k: Union[int, None]
    p: Union[float, None]
    seed: Union[float, None]
    stop_sequences: Union[List[str], None]
    frequency_penalty: Union[float, None]
    presence_penalty: Union[float, None]
    raw_prompting: Union[bool, None]
    request_options: Union[RequestOptions, None]


def _chat_options_dict(options: CohereTextGenOptions) -> dict:
    opt = {
        attr: options.get(attr)
        for attr in CohereTextGenOptions.__annotations__
        if attr not in ["system_prompt"]
    }

    if not opt.get("model") or opt.get("model") == OMIT:
        opt["model"] = "command-r-plus"

    if options.get("system_prompt"):
        opt["preamble"] = options.get("system_prompt")

    # Having None in params will cause JSON exceptions when calling the Cohere API.
    # Hence removing the keys which have None value
    for attr in CohereTextGenOptions.__annotations__:
        if not opt.get(attr) and attr in opt:
            del opt[attr]

    return opt
