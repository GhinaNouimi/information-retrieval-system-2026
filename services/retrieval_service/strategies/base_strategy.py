from abc import ABC, abstractmethod


class RetrievalStrategy(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int):
        pass