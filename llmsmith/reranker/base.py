from abc import ABC, abstractmethod
from typing import List


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, docs: List[str]) -> List[str]:
        pass
