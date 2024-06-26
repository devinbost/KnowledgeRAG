import numpy as np
from abc import ABC, abstractmethod
from pandas import DataFrame
print("importing indexing_strategy file")
class IndexingStrategy(ABC):
    @abstractmethod
    async def add(self, embeddings: np.ndarray, metadata: list) -> None:
        pass

    @abstractmethod
    async def add_batch(self, embeddings_of_batch, dataframe_of_batch: DataFrame) -> None:
        pass

    @abstractmethod
    async def search(self, query_embedding: np.ndarray, limit: int) -> list:
        pass

    @abstractmethod
    async def batch_search(self, source_df: DataFrame) -> list:
        pass

    @abstractmethod
    async def clear(self):
        """Deletes all contents of index."""
        pass