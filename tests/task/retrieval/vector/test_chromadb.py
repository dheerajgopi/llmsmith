from typing import List
import unittest
from unittest import mock

from chromadb import QueryResult

from llmsmith.task.models import TaskInput
from llmsmith.task.retrieval.vector.chromadb import ChromaDBRetriever


class ChromaDBRetrieverTest(unittest.IsolatedAsyncioTestCase):
    @mock.patch("llmsmith.task.retrieval.vector.chromadb.Collection")
    async def test_execute_with_default_doc_processor(self, mock_collection):
        mock_collection.query.return_value = QueryResult(
            documents=[["retrieved_doc1", "retrieved_doc2"]]
        )
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
        assert output.content == "retrieved_doc1\n---\nretrieved_doc2"

    @mock.patch("llmsmith.task.retrieval.vector.chromadb.Collection")
    async def test_execute_with_custom_doc_processor(self, mock_collection):
        mock_collection.query.return_value = QueryResult(
            documents=[["retrieved_doc1", "retrieved_doc2"]]
        )
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
        assert (
            output.content
            == "[document 0] - retrieved_doc1\n\n[document 1] - retrieved_doc2"
        )

    @mock.patch("llmsmith.task.retrieval.vector.chromadb.Collection")
    async def test_execute_with_reranker(self, mock_collection):
        mock_reranker = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.retrieval.vector.chromadb.Reranker",
            side_effect=mock_reranker,
        )
        mock_collection.query.return_value = QueryResult(
            documents=[["retrieved_doc1", "retrieved_doc2"]]
        )
        mock_reranker.rerank.return_value = ["retrieved_doc2", "retrieved_doc1"]
        retriever = ChromaDBRetriever(
            name="test",
            collection=mock_collection,
            embedding_func=lambda x: [[1]],
            reranker=mock_reranker,
        )

        output = await retriever.execute(TaskInput("query"))

        mock_collection.query.assert_called_with(
            query_embeddings=[[1]],
            include=["metadatas", "documents", "distances"],
            n_results=10,
            where=None,
            where_document=None,
        )
        mock_reranker.rerank.assert_called_with(
            query="query", docs=["retrieved_doc1", "retrieved_doc2"]
        )
        assert output.content == "retrieved_doc2\n---\nretrieved_doc1"

    def _chroma_doc_proc_func(self, docs: List[str]) -> str:
        processed_docs = []
        for idx, doc in enumerate(docs):
            processed_docs.append(f"[document {idx}] - {doc}")

        return "\n\n".join(processed_docs)
