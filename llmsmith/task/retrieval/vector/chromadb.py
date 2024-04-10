import logging
from typing import Callable

from llmsmith.task.base import Task
from llmsmith.task.models import TaskInput, TaskOutput
from llmsmith.task.retrieval.vector.base import EmbeddingFunc
from llmsmith.task.retrieval.vector.options.chromadb import (
    ChromaDBQueryOptions,
    _query_options_dict,
)


try:
    from chromadb import Collection, QueryResult
except ImportError:
    raise ImportError(
        "The 'chromadb-client' library is required to use ChromaDBRetriever. You can install it with `pip install \"llmsmith[chromadb]\"`"
    )


log = logging.getLogger(__name__)

# Default options for querying a Qdrant collection.
default_options: ChromaDBQueryOptions = ChromaDBQueryOptions(n_results=10)


def default_doc_processor(res: QueryResult) -> str:
    """
    Formats the retrieved chromadb documents into below format.

    ``
    [0] document-0-content

    [1] document-1-content

    ...

    [n] document-n-content
    ``
    """

    processed_docs = []
    for idx, doc in enumerate(res["documents"][0]):
        processed_docs.append(f"[{idx}] {doc}")

    return "\n\n".join(processed_docs)


class ChromaDBRetriever(Task[str, str]):
    """
    Task for retrieving documents from a collection in ChromaDB.

    :param name: The name of the task.
    :type name: str
    :param collection: The collection to retrieve documents from.
    :type collection: :class:`chromadb.Collection`
    :param embedding_func: Embedding function
    :type embedding_func: :class:`llmsmith.task.retrieval.vector.base.EmbeddingFunc`
    :param doc_processing_func: The function to process the query result, defaults to `llmsmith.task.retrieval.vector.chromadb.default_doc_processor`.
    :type doc_processing_func: Callable[[QueryResult], str], optional
    :param query_options: A dictionary of options to pass to the ChromaDB collection client for querying.
    :type query_options: :class:`llmsmith.task.retrieval.vector.options.chromadb.ChromaDBQueryOptions`, optional
    """

    def __init__(
        self,
        name: str,
        collection: Collection,
        embedding_func: EmbeddingFunc,
        doc_processing_func: Callable[[QueryResult], str] = default_doc_processor,
        query_options: ChromaDBQueryOptions = default_options,
    ) -> None:
        super().__init__(name)

        if not embedding_func:
            raise ValueError("Embedding function ('embedding_func') is required")

        self.collection = collection
        self.embedding_func = embedding_func
        self.doc_processing_func = doc_processing_func
        self.query_options = query_options

    async def execute(self, task_input: TaskInput[str]) -> TaskOutput[str]:
        """
        Executes the task of retrieving documents from the chromadb collection.

        :param task_input: The input for the task.
        :type task_input: :class:`llmsmith.task.models.TaskInput[str]`
        :return: The output of the task, which includes the processed result and the raw output from chromadb.
        :rtype: :class:`llmsmith.task.models.TaskOutput[str]`
        """
        query_options: dict = _query_options_dict(self.query_options)

        log.debug(
            f"ChromaDB query request: input string: {task_input.content}\n OPTIONS: {query_options}"
        )

        embeddings = self.embedding_func([task_input.content])

        res: QueryResult = self.collection.query(
            query_embeddings=embeddings,
            include=["metadatas", "documents", "distances"],
            **query_options,
        )

        processed_res: str = self.doc_processing_func(res)
        return TaskOutput(content=processed_res, raw_output=res)
