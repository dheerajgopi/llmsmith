from typing import Callable, List, Union

# Type alias for embedding function
EmbeddingFunc = Callable[[List[str]], List[List[Union[float, int]]]]


def default_doc_processor(docs: List[str]) -> str:
    """
    Formats the retrieved documents into below format.

    ``
    document-0-content
    ---
    document-1-content
    ---
    ...
    ---
    document-n-content
    ``
    """

    return "\n---\n".join([doc for doc in docs])
