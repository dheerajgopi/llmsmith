from typing import Dict, List, TypedDict, Union

try:
    from pinecone import SparseValues
except ImportError:
    raise ImportError(
        "The 'pinecone-client' library is required to use Pinecone. You can install it with `pip install \"llmsmith[pinecone]\"`"
    )


class PineconeQueryOptions(TypedDict):
    """
    A dictionary of options to pass while querying a Pinecone index.
    The option names are same as the ones used in `query` method of Pinecone `Index` client.
    Refer below links for more info.

    * https://sdk.pinecone.io/python/pinecone.html#query-an-index
    """

    top_k: Union[int, None]
    namespace: Union[str, None]
    filter: Union[Dict[str, Union[str, float, int, bool, List, dict]], None]
    include_values: Union[bool, None]
    sparse_vector: Union[SparseValues, Dict[str, Union[List[float], List[int]]], None]


def _query_options_dict(options: PineconeQueryOptions) -> dict:
    opt = {attr: options.get(attr) for attr in PineconeQueryOptions.__annotations__}

    if not opt.get("top_k"):
        opt["top_k"] = 10

    if not opt.get("include_values"):
        opt["include_values"] = False

    return opt
