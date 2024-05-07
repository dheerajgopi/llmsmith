from typing import Sequence, TypedDict, Union

try:
    from qdrant_client.conversions.common_types import (
        Filter,
        SearchParams,
        ReadConsistency,
        ShardKeySelector,
    )
except ImportError:
    raise ImportError(
        "The 'qdrant-client' library is required to use Qdrant. You can install it with `pip install \"llmsmith[qdrant]\"`"
    )


class QdrantQueryOptions(TypedDict):
    """
    A dictionary of options to pass while querying a qdrant collection.
    The option names are same as the ones used in `search` method of AsyncQdrantClient.
    Refer below links for more info.

    * https://python-client.qdrant.tech/qdrant_client.async_qdrant_client
    """

    query_filter: Union[Filter, None]
    search_params: Union[SearchParams, None]
    limit: Union[int, None]
    offset: Union[int, None]
    with_vectors: Union[bool, Sequence[str], None]
    score_threshold: Union[float, None]
    consistency: Union[ReadConsistency, None]
    shard_key_selector: Union[ShardKeySelector, None]
    timeout: Union[int, None]


def _query_options_dict(options: QdrantQueryOptions) -> dict:
    opt = {attr: options.get(attr) for attr in QdrantQueryOptions.__annotations__}
    opt["with_payload"] = True

    if not opt.get("limit"):
        opt["limit"] = 10

    return opt
