import unittest
from unittest import mock

from chromadb import QueryResult

from llmsmith.task.models import TaskInput
from llmsmith.task.retrieval.vector.chromadb import ChromaDBRetriever


class ChromaDBRetrieverTest(unittest.IsolatedAsyncioTestCase):
    @mock.patch("llmsmith.task.retrieval.vector.chromadb.Collection")
    async def test_execute_with_default_doc_processor(self, mock_collection):
        mock_collection.query.return_value = QueryResult(documents=[["retrieved_doc"]])
        retriever = ChromaDBRetriever(
            name="test", collection=mock_collection, embedding_func=lambda x: [[1]]
        )

        output = await retriever.execute(TaskInput("query"))

        mock_collection.query.assert_called_with(
            query_embeddings=[[1]],
            include=["metadatas", "documents", "distances"],
            n_results=10,
            where=None,
            where_document=None,
        )
        assert output.content == "[0] retrieved_doc"

    @mock.patch("llmsmith.task.retrieval.vector.chromadb.Collection")
    async def test_execute_with_custom_doc_processor(self, mock_collection):
        mock_collection.query.return_value = QueryResult(documents=[["retrieved_doc"]])
        retriever = ChromaDBRetriever(
            name="test",
            collection=mock_collection,
            embedding_func=lambda x: [[1]],
            doc_processing_func=self._chroma_doc_proc_func,
        )

        output = await retriever.execute(TaskInput("query"))

        mock_collection.query.assert_called_with(
            query_embeddings=[[1]],
            include=["metadatas", "documents", "distances"],
            n_results=10,
            where=None,
            where_document=None,
        )
        assert output.content == "[document 0] - retrieved_doc"

    def _chroma_doc_proc_func(self, res: QueryResult) -> str:
        processed_docs = []
        for idx, doc in enumerate(res["documents"][0]):
            processed_docs.append(f"[document {idx}] - {doc}")

        return "\n\n".join(processed_docs)
