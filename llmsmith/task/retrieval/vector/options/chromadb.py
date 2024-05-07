from typing import TypedDict

try:
    from chromadb import Where, WhereDocument
except ImportError:
    raise ImportError(
        "The 'chromadb-client' library is required to use ChromaDB. You can install it with `pip install \"llmsmith[chromadb]\"`"
    )


class ChromaDBQueryOptions(TypedDict):
    """
    A dictionary of options to pass while querying a Chroma DB collection.
    The option names are same as the ones used in `query` method of ChromaDB `Collection` client.
    Refer below links for more info.

    * https://docs.trychroma.com/usage-guide#querying-a-collection
    """

    n_results: int
    where: Where
    where_document: WhereDocument


def _query_options_dict(options: ChromaDBQueryOptions) -> dict:
    opt = {attr: options.get(attr) for attr in ChromaDBQueryOptions.__annotations__}

    if not opt.get("n_results"):
        opt["n_results"] = 10

    return opt
