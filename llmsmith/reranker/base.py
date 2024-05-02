from abc import ABC, abstractmethod
from typing import List


class Reranker(ABC):
    """
    Abstract base class for document rerankers.
    """

    @abstractmethod
    async def rerank(self, query: str, docs: List[str]) -> List[str]:
        """
        Rerank documents and return the same.

        :param query: query used to retrieve documents from vector DB.
        :type query: str
        :param docs: list of documents retrieved from vector DB.
        :type docs: List[str]
        :return: reranked documents
        :rtype: List[str]
        """
        pass
