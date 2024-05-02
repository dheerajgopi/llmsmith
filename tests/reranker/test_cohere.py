import unittest
from unittest import mock

from cohere.types.rerank_response import RerankResponse
from cohere.types.rerank_response_results_item import (
    RerankResponseResultsItem,
)

from llmsmith.reranker.cohere import CohereReranker
from llmsmith.reranker.options.cohere import CohereRerankerOptions


class CohereRerankerTest(unittest.IsolatedAsyncioTestCase):
    async def test_rerank_with_default_options(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.reranker.cohere.cohere.AsyncClient",
            side_effect=mock_client,
        )

        mock_client.rerank.return_value = RerankResponse(
            id="1",
            results=[
                RerankResponseResultsItem(index=1, relevance_score=1),
                RerankResponseResultsItem(index=0, relevance_score=0.9),
            ],
        )
        reranker = CohereReranker(mock_client)
        input_docs = ["doc1", "doc2"]

        output = await reranker.rerank("query", input_docs)

        mock_client.rerank.assert_called_with(
            query="query",
            documents=input_docs,
            model="rerank-english-v2.0",
            top_n=None,
            rank_fields=None,
            return_documents=None,
            max_chunks_per_doc=None,
            request_options=None,
        )

        assert output == ["doc2", "doc1"]

    async def test_rerank_with_custom_options(self):
        mock_client = mock.AsyncMock()
        mock.patch(
            "llmsmith.reranker.cohere.cohere.AsyncClient",
            side_effect=mock_client,
        )

        mock_client.rerank.return_value = RerankResponse(
            id="1",
            results=[
                RerankResponseResultsItem(index=1, relevance_score=1),
                RerankResponseResultsItem(index=0, relevance_score=0.9),
            ],
        )
        reranker = CohereReranker(
            mock_client, CohereRerankerOptions(model="rerank-english-v3.0", top_n=5)
        )
        input_docs = ["doc1", "doc2"]

        output = await reranker.rerank("query", input_docs)

        mock_client.rerank.assert_called_with(
            query="query",
            documents=input_docs,
            model="rerank-english-v3.0",
            top_n=5,
            rank_fields=None,
            return_documents=None,
            max_chunks_per_doc=None,
            request_options=None,
        )

        assert output == ["doc2", "doc1"]
