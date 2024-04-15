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
            ScoredPoint(id=1, version=1, score=1.0, payload={"doc": "retrieved_doc"})
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

        assert output.content == "[0] retrieved_doc"

    async def test_execute_with_custom_doc_processor(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.task.retrieval.vector.qdrant.AsyncQdrantClient",
            side_effect=mock_client,
        )

        mock_client.search.return_value = [
            ScoredPoint(id=1, version=1, score=1.0, payload={"doc": "retrieved_doc"})
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

        assert output.content == "document[0] - retrieved_doc"

    def _qdrant_custom_doc_processor(self, res: List[ScoredPoint]) -> str:
        processed_docs = []
        for idx, doc in enumerate(res):
            processed_docs.append(f"document[{idx}] - {doc.payload.get('doc', '')}")

        return "\n\n".join(processed_docs)
