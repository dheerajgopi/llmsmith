from typing import List, TypedDict, Union

try:
    from cohere.core.request_options import RequestOptions
except ImportError:
    raise ImportError(
        "The 'cohere' library is required to use Cohere reranker. You can install it with `pip install \"llmsmith[cohere]\"`"
    )


class CohereRerankerOptions(TypedDict):
    """
    A dictionary of options to be passed into Cohere reranker.
    The option names are same as the ones used in the reranker method of Cohere client.
    Refer below links for more info.

    * https://docs.cohere.com/reference/rerank
    """

    model: str
    top_n: Union[int, None]
    rank_fields: Union[List[str], None]
    return_documents: Union[bool, None]
    max_chunks_per_doc: Union[int, None]
    request_options: Union[RequestOptions, None]


def _rerank_options_dict(options: CohereRerankerOptions = {}) -> dict:
    if not options:
        options = {}

    opt = {attr: options.get(attr) for attr in CohereRerankerOptions.__annotations__}

    if not opt.get("model"):
        opt["model"] = "rerank-english-v2.0"

    return opt
