import logging
from typing import Callable, List, Union

from llmsmith.task.base import Task
from llmsmith.task.models import TaskInput, TaskOutput
from llmsmith.task.retrieval.vector.base import EmbeddingFunc
from llmsmith.task.retrieval.vector.options.qdrant import (
    QdrantQueryOptions,
    _query_options_dict,
)


try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.conversions.common_types import ScoredPoint
except ImportError:
    raise ImportError(
        "The 'qdrant-client' library is required to use QdrantRetriever. You can install it with `pip install \"llmsmith[qdrant]\"`"
    )


log = logging.getLogger(__name__)

# Default options for querying a Qdrant collection.
default_options: QdrantQueryOptions = QdrantQueryOptions(limit=10, with_vectors=False)


def default_doc_processor_func(field: str):
    def default_doc_processor(res: List[ScoredPoint]) -> str:
        """
        Formats the retrieved qdrant documents into below format.

        ``
        [0] document-0-content

        [1] document-1-content

        ...

        [n] document-n-content
        ``
        """

        processed_docs = []
        for idx, doc in enumerate(res):
            processed_docs.append(f"[{idx}] {doc.payload.get(field, '')}")

        return "\n\n".join(processed_docs)

    return default_doc_processor


class QdrantRetriever(Task[str, str]):
    """
    Task for retrieving documents from a collection in Qdrant.

    :param name: The name of the task.
    :type name: str
    :param client: Qdrant client.
    :type client: :class:`qdrant_client.AsyncQdrantClient`
    :param collection_name: Qdrant collection name.
    :type collection_name: str
    :param embedding_func: Embedding function
    :type embedding_func: :class:`llmsmith.task.retrieval.vector.base.EmbeddingFunc`
    :param embedded_field_name: name of the field in the document on which embeddedings are created while uploading data to the Qdrant collection
    :type embedded_field_name: str
    :param doc_processing_func: The function to process the query result.
    :type doc_processing_func: Callable[[List[ScoredPoint]], str], optional
    :param query_options: A dictionary of options to pass to the Qdrant client for querying.
    :type query_options: :class:`llmsmith.task.retrieval.vector.options.qdrant.QdrantQueryOptions`, optional
    """

    def __init__(
        self,
        name: str,
        client: AsyncQdrantClient,
        collection_name: str,
        embedding_func: EmbeddingFunc,
        embedded_field_name: str,
        doc_processing_func: Union[Callable[[List[ScoredPoint]], str], None] = None,
        query_options: QdrantQueryOptions = default_options,
    ) -> None:
        super().__init__(name)

        if not embedding_func:
            raise ValueError("Embedding function ('embedding_func') is required")

        self.client = client
        self.collection_name = collection_name
        self.embedding_func = embedding_func
        self.embedded_field_name = embedded_field_name
        self.query_options = query_options

        if not doc_processing_func:
            self.doc_processing_func = default_doc_processor_func(embedded_field_name)
        else:
            self.doc_processing_func = doc_processing_func

    async def execute(self, task_input: TaskInput[str]) -> TaskOutput[str]:
        """
        Executes the task of retrieving documents from the qdrant collection.

        :param task_input: The input for the task.
        :type task_input: :class:`llmsmith.task.models.TaskInput[str]`
        :return: The output of the task, which includes the processed result and the raw output from Qdrant.
        :rtype: :class:`llmsmith.task.models.TaskOutput[str]`
        """

        query_options: dict = _query_options_dict(self.query_options)

        log.debug(
            f"Qdrant query request: input string: {task_input.content}\n OPTIONS: {query_options}"
        )
        embeddings = self.embedding_func([task_input.content])

        res = await self.client.search(
            collection_name=self.collection_name,
            query_vector=embeddings[0],
            **query_options,
        )
        processed_res: str = self.doc_processing_func(res)

        return TaskOutput(content=processed_res, raw_output=res)
