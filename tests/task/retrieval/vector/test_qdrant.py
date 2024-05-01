from typing import List
import unittest
from unittest import mock

from qdrant_client.conversions.common_types import ScoredPoint

from llmsmith.task.models import TaskInput
from llmsmith.task.retrieval.vector.qdrant import QdrantRetriever


class QdrantRetrieverTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_with_default_doc_processor(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.retrieval.vector.qdrant.AsyncQdrantClient",
            side_effect=mock_client,
        )

        mock_client.search.return_value = [
            ScoredPoint(id=1, version=1, score=1.0, payload={"doc": "retrieved_doc1"}),
            ScoredPoint(id=2, version=1, score=0.9, payload={"doc": "retrieved_doc2"}),
        ]
        retriever = QdrantRetriever(
            name="test",
            client=mock_client,
            collection_name="test_collection",
            embedding_func=lambda x: [[1]],
            embedded_field_name="doc",
        )

        output = await retriever.execute(TaskInput("query"))

        mock_client.search.assert_called_with(
            collection_name="test_collection",
            query_vector=[1],
            query_filter=None,
            search_params=None,
            limit=10,
            offset=None,
            with_vectors=False,
            score_threshold=None,
            consistency=None,
            shard_key_selector=None,
            timeout=None,
            with_payload=True,
        )

        assert output.content == "retrieved_doc1\n---\nretrieved_doc2"

    async def test_execute_with_custom_doc_processor(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.retrieval.vector.qdrant.AsyncQdrantClient",
            side_effect=mock_client,
        )

        mock_client.search.return_value = [
            ScoredPoint(id=1, version=1, score=1.0, payload={"doc": "retrieved_doc1"}),
            ScoredPoint(id=2, version=1, score=0.9, payload={"doc": "retrieved_doc2"}),
        ]
        retriever = QdrantRetriever(
            name="test",
            client=mock_client,
            collection_name="test_collection",
            embedding_func=lambda x: [[1]],
            embedded_field_name="doc",
            doc_processing_func=self._qdrant_custom_doc_processor,
        )

        output = await retriever.execute(TaskInput("query"))

        mock_client.search.assert_called_with(
            collection_name="test_collection",
            query_vector=[1],
            query_filter=None,
            search_params=None,
            limit=10,
            offset=None,
            with_vectors=False,
            score_threshold=None,
            consistency=None,
            shard_key_selector=None,
            timeout=None,
            with_payload=True,
        )

        assert (
            output.content
            == "[document 0] - retrieved_doc1\n\n[document 1] - retrieved_doc2"
        )

    async def test_execute_with_reranker(self):
        mock_client = mock.AsyncMock()
        mock_reranker = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.retrieval.vector.qdrant.AsyncQdrantClient",
            side_effect=mock_client,
        )
        mock.patch(
            "llmsmith.task.retrieval.vector.qdrant.Reranker",
            side_effect=mock_reranker,
        )

        mock_client.search.return_value = [
            ScoredPoint(id=1, version=1, score=1.0, payload={"doc": "retrieved_doc1"}),
            ScoredPoint(id=2, version=1, score=0.9, payload={"doc": "retrieved_doc2"}),
        ]
        mock_reranker.rerank.return_value = ["retrieved_doc2", "retrieved_doc1"]

        retriever = QdrantRetriever(
            name="test",
            client=mock_client,
            collection_name="test_collection",
            embedding_func=lambda x: [[1]],
            embedded_field_name="doc",
            reranker=mock_reranker,
        )

        output = await retriever.execute(TaskInput("query"))

        mock_client.search.assert_called_with(
            collection_name="test_collection",
            query_vector=[1],
            query_filter=None,
            search_params=None,
            limit=10,
            offset=None,
            with_vectors=False,
            score_threshold=None,
            consistency=None,
            shard_key_selector=None,
            timeout=None,
            with_payload=True,
        )
        mock_reranker.rerank.assert_called_with(
            query="query", docs=["retrieved_doc1", "retrieved_doc2"]
        )

        assert output.content == "retrieved_doc2\n---\nretrieved_doc1"

    def _qdrant_custom_doc_processor(self, docs: List[str]) -> str:
        processed_docs = []
        for idx, doc in enumerate(docs):
            processed_docs.append(f"[document {idx}] - {doc}")

        return "\n\n".join(processed_docs)
