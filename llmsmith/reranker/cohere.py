import logging
from typing import List, Union

from llmsmith.reranker.base import Reranker
from llmsmith.reranker.options.cohere import CohereRerankerOptions, _rerank_options_dict

try:
    import cohere
    from cohere.types.rerank_request_documents_item import (
        RerankRequestDocumentsItemText,
    )
    from cohere.types.rerank_response import RerankResponse
except ImportError:
    raise ImportError(
        "The 'cohere' library is required to use Cohere reranker. You can install it with `pip install \"llmsmith[cohere]\"`"
    )


log = logging.getLogger(__name__)


class CohereReranker(Reranker):
    """
    Reranker based on Cohere reranking.

    :param client: An instance of the Cohere client.
    :type client: :class:`cohere.AsyncClient`
    :param options: A dictionary of options to pass to the rerank method of Cohere client.
    :type options: :class:`llmsmith.reranker.options.cohere.CohereRerankerOptions`, optional
    """

    def __init__(
        self,
        client: cohere.AsyncClient,
        options: Union[CohereRerankerOptions, None] = None,
    ) -> None:
        self.client = client
        self._options = options or {}

    async def rerank(self, query: str, docs: List[str]) -> List[str]:
        """
        Rerank documents using Cohere API.

        :param query: Query used to fetch documents from vector database.
        :type query: str
        :param docs: Documents returned by vector database.
        :type docs: List[str]
        :returns: Reranked documents returned by Cohere.
        :rtype: List[str]
        """
        return await self.__rerank_docs(query, docs)

    async def __rerank_docs(
        self, query: str, docs: List[Union[str, RerankRequestDocumentsItemText]]
    ) -> List[Union[str, RerankRequestDocumentsItemText]]:
        """
        Rerank documents using Cohere API.

        :param query: Query used to fetch documents from vector database.
        :type query: str
        :param docs: Documents returned by vector database.
        :type docs: List[Union[str, :class:`cohere.types.rerank_request_documents_item.RerankRequestDocumentsItemText`]]
        :returns: Reranked documents returned by Cohere.
        :rtype: List[Union[str, :class:`cohere.types.rerank_request_documents_item.RerankRequestDocumentsItemText`]]
        """

        rerank_options: dict = _rerank_options_dict(self._options)

        log.debug(f"CohereReranker input docs: {docs}")

        rerank_res: RerankResponse = await self.client.rerank(
            query=query, documents=docs, **rerank_options
        )

        reranked_docs: List[Union[str, RerankRequestDocumentsItemText]] = []

        for each in rerank_res.results:
            reranked_docs.append(docs[each.index])

        log.debug(f"CohereReranker reranked docs: {reranked_docs}")

        return reranked_docs
