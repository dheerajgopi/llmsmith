from typing import List
import unittest
from unittest import mock

from pinecone import QueryResponse
from pinecone.core.client.model.scored_vector import ScoredVector

from llmsmith.task.models import TaskInput
from llmsmith.task.retrieval.vector.options.pinecone import PineconeQueryOptions
from llmsmith.task.retrieval.vector.pinecone import PineconeRetriever
from llmsmith.task.retrieval.vector.qdrant import QdrantRetriever


class PineconeRetrieverTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_with_default_doc_processor(self):
        mock_index = mock.Mock()
        mock.patch(
            "llmsmith.task.retrieval.vector.pinecone.Index",
            side_effect=mock_index,
        )

        mock_index.query.return_value = QueryResponse(matches=[
            ScoredVector(id="id1", metadata={"doc": "retrieved_doc1"}), 
            ScoredVector(id="id2", metadata={"doc": "retrieved_doc2"})]
        )
        retriever = PineconeRetriever(
            name="test",
            index=mock_index,
            embedding_func=lambda x: [[1]],
            text_field_name="doc",
        )

        output = await retriever.execute(TaskInput("query"))

        mock_index.query.assert_called_with(
            vector=[[1]],
            include_metadata=True, 
            top_k=10, 
            namespace=None, 
            filter=None, 
            include_values=False, 
            sparse_vector=None
        )

        assert output.content == "retrieved_doc1\n---\nretrieved_doc2"
    
    async def test_execute_with_custom_query_options(self):
        mock_index = mock.Mock()
        mock.patch(
            "llmsmith.task.retrieval.vector.pinecone.Index",
            side_effect=mock_index,
        )

        mock_index.query.return_value = QueryResponse(matches=[
            ScoredVector(id="id1", metadata={"doc": "retrieved_doc1"}), 
            ScoredVector(id="id2", metadata={"doc": "retrieved_doc2"})]
        )
        retriever = PineconeRetriever(
            name="test",
            index=mock_index,
            embedding_func=lambda x: [[1]],
            text_field_name="doc",
            query_options=PineconeQueryOptions(namespace="ns", top_k=5)
        )

        output = await retriever.execute(TaskInput("query"))

        mock_index.query.assert_called_with(
            vector=[[1]],
            include_metadata=True, 
            top_k=5, 
            namespace="ns", 
            filter=None, 
            include_values=False, 
            sparse_vector=None
        )

        assert output.content == "retrieved_doc1\n---\nretrieved_doc2"

    async def test_execute_with_custom_doc_processor(self):
        mock_index = mock.Mock()
        mock.patch(
            "llmsmith.task.retrieval.vector.pinecone.Index",
            side_effect=mock_index,
        )

        mock_index.query.return_value = QueryResponse(matches=[
            ScoredVector(id="id1", metadata={"doc": "retrieved_doc1"}), 
            ScoredVector(id="id2", metadata={"doc": "retrieved_doc2"})]
        )
        retriever = PineconeRetriever(
            name="test",
            index=mock_index,
            embedding_func=lambda x: [[1]],
            text_field_name="doc",
            doc_processing_func=self._pinecone_custom_doc_processor,
        )

        output = await retriever.execute(TaskInput("query"))

        mock_index.query.assert_called_with(
            vector=[[1]],
            include_metadata=True, 
            top_k=10, 
            namespace=None, 
            filter=None, 
            include_values=False, 
            sparse_vector=None
        )

        assert (
            output.content
            == "[document 0] - retrieved_doc1\n\n[document 1] - retrieved_doc2"
        )

    async def test_execute_with_reranker(self):
        mock_index = mock.Mock()
        mock_reranker = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.retrieval.vector.pinecone.Index",
            side_effect=mock_index,
        )

        mock_index.query.return_value = QueryResponse(matches=[
            ScoredVector(id="id1", metadata={"doc": "retrieved_doc1"}), 
            ScoredVector(id="id2", metadata={"doc": "retrieved_doc2"})]
        )
        mock.patch(
            "llmsmith.task.retrieval.vector.pinecone.Reranker",
            side_effect=mock_reranker,
        )
        mock_reranker.rerank.return_value = ["retrieved_doc2", "retrieved_doc1"]

        retriever = PineconeRetriever(
            name="test",
            index=mock_index,
            embedding_func=lambda x: [[1]],
            text_field_name="doc",
            reranker=mock_reranker
        )

        output = await retriever.execute(TaskInput("query"))

        mock_index.query.assert_called_with(
            vector=[[1]],
            include_metadata=True, 
            top_k=10, 
            namespace=None, 
            filter=None, 
            include_values=False, 
            sparse_vector=None
        )

        mock_reranker.rerank.assert_called_with(
            query="query", docs=["retrieved_doc1", "retrieved_doc2"]
        )

        assert output.content == "retrieved_doc2\n---\nretrieved_doc1"

    def _pinecone_custom_doc_processor(self, docs: List[str]) -> str:
        processed_docs = []
        for idx, doc in enumerate(docs):
            processed_docs.append(f"[document {idx}] - {doc}")

        return "\n\n".join(processed_docs)
