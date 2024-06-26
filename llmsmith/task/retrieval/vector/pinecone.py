import logging
from typing import Callable, List

from llmsmith.reranker.base import Reranker
from llmsmith.task.base import Task
from llmsmith.task.models import TaskInput, TaskOutput
from llmsmith.task.retrieval.vector.base import EmbeddingFunc, default_doc_processor
from llmsmith.task.retrieval.vector.options.pinecone import (
    PineconeQueryOptions,
    _query_options_dict,
)


try:
    from pinecone import Index, QueryResponse
except ImportError:
    raise ImportError(
        "The 'pinecone-client' library is required to use Pinecone. You can install it with `pip install \"llmsmith[pinecone]\"`"
    )


log = logging.getLogger(__name__)

# Default options for querying a Pinecone index.
default_options: PineconeQueryOptions = PineconeQueryOptions(top_k=10)


class BasePineconeTask(Task[str, List[str]]):
    """
    Task for retrieving documents from an index in Pinecone.

    :param name: The name of the task.
    :type name: str
    :param index: The index to retrieve documents from.
    :type index: :class:`pinecone.Index`
    :param embedding_func: Embedding function
    :type embedding_func: :class:`llmsmith.task.retrieval.vector.base.EmbeddingFunc`
    :param text_field_name: name of the field in the metadata to be fetched during the retrieval
    :type text_field_name: str
    :param query_options: A dictionary of options to pass to the Pinecone index for querying.
    :type query_options: :class:`llmsmith.task.retrieval.vector.options.pinecone.PineconeQueryOptions`, optional
    :param reranker: Rerank the documents based on the query used to retrieve the documents.
    :type reranker: :class:`llmsmith.reranker.base.Reranker`, optional
    """

    def __init__(
        self,
        name: str,
        index: Index,
        embedding_func: EmbeddingFunc,
        text_field_name: str,
        query_options: PineconeQueryOptions = default_options,
        reranker: Reranker = None,
    ) -> None:
        super().__init__(name)

        if not embedding_func:
            raise ValueError("Embedding function ('embedding_func') is required")

        self.index = index
        self.embedding_func = embedding_func
        self.text_field_name = text_field_name
        self.query_options = query_options
        self._reranker = reranker

    async def execute(self, task_input: TaskInput[str]) -> TaskOutput[List[str]]:
        """
        Executes the task of retrieving documents from the Pinecone collection.

        :param task_input: The input for the task.
        :type task_input: :class:`llmsmith.task.models.TaskInput[str]`
        :return: The output of the task, which includes the processed result and the raw output from Pinecone.
        :rtype: :class:`llmsmith.task.models.TaskOutput[List[str]]`
        """
        query_options: dict = _query_options_dict(self.query_options)

        log.debug(
            f"Pinecone query request: input string: {task_input.content}\n OPTIONS: {query_options}"
        )

        embeddings = self.embedding_func([task_input.content])

        res: QueryResponse = self.index.query(
            vector=embeddings, include_metadata=True, **query_options
        )

        docs = [
            doc.get("metadata", {}).get(self.text_field_name) for doc in res["matches"]
        ]
        if self._reranker:
            docs = await self._reranker.rerank(query=task_input.content, docs=docs)

        return TaskOutput(content=docs, raw_output=res)


class PineconeRetriever(Task[str, str]):
    """
    Task for retrieving documents from an index in Pinecone.

    :param name: The name of the task.
    :type name: str
    :param index: The index to retrieve documents from.
    :type index: :class:`pinecone.Index`
    :param embedding_func: Embedding function
    :type embedding_func: :class:`llmsmith.task.retrieval.vector.base.EmbeddingFunc`
    :param text_field_name: name of the field in the metadata to be fetched during the retrieval
    :type text_field_name: str
    :param doc_processing_func: The function to process the query result, defaults to `llmsmith.task.retrieval.vector.base.default_doc_processor`.
    :type doc_processing_func: Callable[[List[str]], str], optional
    :param query_options: A dictionary of options to pass to the Pinecone index for querying.
    :type query_options: :class:`llmsmith.task.retrieval.vector.options.pinecone.PineconeQueryOptions`, optional
    :param reranker: Rerank the documents based on the query used to retrieve the documents.
    :type reranker: :class:`llmsmith.reranker.base.Reranker`, optional
    """

    def __init__(
        self,
        name: str,
        index: Index,
        embedding_func: EmbeddingFunc,
        text_field_name: str,
        doc_processing_func: Callable[[List[str]], str] = default_doc_processor,
        query_options: PineconeQueryOptions = default_options,
        reranker: Reranker = None,
    ) -> None:
        super().__init__(name)

        self._task = BasePineconeTask(
            name=name,
            index=index,
            embedding_func=embedding_func,
            text_field_name=text_field_name,
            query_options=query_options,
            reranker=reranker,
        )
        self.doc_processing_func = doc_processing_func

    async def execute(self, task_input: TaskInput[str]) -> TaskOutput[str]:
        """
        Executes the task of retrieving documents from the Pinecone collection.

        :param task_input: The input for the task.
        :type task_input: :class:`llmsmith.task.models.TaskInput[str]`
        :return: The output of the task, which includes the processed result and the raw output from Pinecone.
        :rtype: :class:`llmsmith.task.models.TaskOutput[str]`
        """
        task_res: TaskOutput[List[str]] = await self._task.execute(task_input)

        processed_res: str = self.doc_processing_func(task_res.content)
        return TaskOutput(content=processed_res, raw_output=task_res.raw_output)
