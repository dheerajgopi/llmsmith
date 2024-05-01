import logging
from typing import List, Union

from llmsmith.reranker.base import Reranker

try:
    import cohere
    from cohere.types.rerank_request_documents_item import (
        RerankRequestDocumentsItemText,
    )
    from cohere.types.rerank_response import RerankResponse
except ImportError:
    raise ImportError(
        "The 'cohere' library is required to use CohereReranker. You can install it with `pip install \"llmsmith[cohere]\"`"
    )


log = logging.getLogger(__name__)


class CohereReranker(Reranker):
    def __init__(self, client: cohere.AsyncClient) -> None:
        self.client = client

    async def rerank(self, query: str, docs: List[str]) -> List[str]:
        return await self.rerank_docs(query, docs)

    async def rerank_docs(
        self, query: str, docs: List[Union[str, RerankRequestDocumentsItemText]]
    ) -> List[Union[str, RerankRequestDocumentsItemText]]:
        log.debug(f"CohereReranker input docs: {docs}")
        rerank_res: RerankResponse = await self.client.rerank(
            model="rerank-english-v2.0",
            query=query,
            documents=docs,
            return_documents=False,
        )

        reranked_docs: List[Union[str, RerankRequestDocumentsItemText]] = []

        for each in rerank_res.results:
            reranked_docs.append(docs[each.index])

        log.debug(f"CohereReranker reranked docs: {reranked_docs}")

        return reranked_docs
