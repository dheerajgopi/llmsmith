from typing import Callable

from llmsmith.task.base import Task
from llmsmith.task.models import TaskInput, TaskOutput


try:
    from chromadb import Collection, QueryResult, Where, WhereDocument
except ImportError:
    raise ImportError(
        "The 'chromadb-client' library is required to use ChromaDBRetriever. You can install it with `pip install llmsmith[chromadb]`"
    )


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
    :param docs_to_retrieve: The number of documents to retrieve, defaults to 5.
    :type docs_to_retrieve: int, optional
    :param where: Filters to be applied on the `metadata` field, defaults to None.
    :type where: :class:`chromadb.Where`, optional
    :param where_doc: Filters to be applied on the document itself, defaults to None.
    :type where_doc: :class:`chromadb.WhereDocument`, optional
    :param doc_processing_func: The function to process the query result, defaults to `llmsmith.task.retrieval.vector.chromadb.default_doc_processor`.
    :type doc_processing_func: Callable[[QueryResult], str], optional
    """

    def __init__(
        self,
        name: str,
        collection: Collection,
        docs_to_retrieve: int = 5,
        where: Where = None,
        where_doc: WhereDocument = None,
        doc_processing_func: Callable[[QueryResult], str] = default_doc_processor,
    ) -> None:
        super().__init__(name)

        self.collection = collection
        self.docs_to_retrieve = docs_to_retrieve
        self.where = where
        self.where_doc = where_doc
        self.doc_processing_func = doc_processing_func

    async def execute(self, task_input: TaskInput[str]) -> TaskOutput[str]:
        """
        Executes the task of retrieving documents from the chromadb collection.

        :param task_input: The input for the task.
        :type task_input: :class:`llmsmith.task.models.TaskInput[str]`
        :return: The output of the task, which includes the processed result and the raw output from chromadb.
        :rtype: :class:`llmsmith.task.models.TaskOutput[str]`
        """
        res: QueryResult = self.collection.query(
            query_texts=[task_input.content],
            n_results=self.docs_to_retrieve,
            include=["metadatas", "documents", "distances"],
            where=self.where,
            where_document=self.where_doc,
        )

        processed_res: str = self.doc_processing_func(res)
        return TaskOutput(content=processed_res, raw_output=res)
