import faiss
import numpy as np
from abc import ABC, abstractmethod
from pandas import DataFrame
from typing import Any, List
import torch
print("Importing faiss_indexing")
from .indexing_strategy import IndexingStrategy
print("Importing faiss_indexing, part 2")
seed = 42
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# For debugging:
# torch.set_printoptions(profile="full")

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
# Concrete Strategy for FAISS Indexing
class FaissIndexing(IndexingStrategy):
    def __init__(self, vector_length: int, index_m: int, efConstruction: int, batch_size: int = 16):
        self.index = faiss.IndexHNSWFlat(vector_length, index_m)
        print("Faiss index dimension:", self.index.d)
        self.index.hnsw.efConstruction = efConstruction
        self.metadata = []
        self.vector_length = vector_length
        self.index_m = index_m
        self.efConstruction = efConstruction
        self.prediction_id_map = {}
        self.added_ids = set()

        self.added_chunks = set()

        self.result_map = {}
        self.batch_size = batch_size

    def populate_result_map(self, row):
        """
        Function to populate the dictionary with (chunk1, chunk2) as key and result as value.
        
        Parameters:
        row (pd.Series): A row of the DataFrame containing chunk1, chunk2, and result.
        """
        chunk1, chunk2, result = row['chunk1'], row['chunk2'], row['result']
        # if chunk1 == chunk2:
        #     print("debug")
        self.result_map[(chunk1, chunk2)] = result
        
    def get_result(self, chunk1: str, chunk2: str):
        """
        Function to get the result from the dictionary given chunk1 and chunk2.
        
        Parameters:
        chunk1 (str): The first chunk.
        chunk2 (str): The second chunk.
        
        Returns:
        str: The result corresponding to the provided chunk1 and chunk2.
        """
        return self.result_map.get((chunk1, chunk2), None)

    async def add_all(self, full_dataframe):
        full_dataframe.apply(lambda row: self.populate_result_map(row), axis=1)
        with torch.no_grad():
            filtered_df = full_dataframe
            
            # Calculate the number of batches
            num_batches = len(filtered_df) // self.batch_size + (0 if len(filtered_df) % self.batch_size == 0 else 1)

            # Process each batch
            for batch_idx in range(num_batches):
                # Slice the DataFrame to get the batch
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_df = filtered_df.iloc[start_idx:end_idx]

                await self.add_batch(batch_df)
        
    #     await self.indexing_strategy.write_batch(ready_to_write)
    async def add_batch(self, dataframe_of_batch: DataFrame) -> None:
        metadata_entries = dataframe_of_batch.apply(
            lambda row: {"_id": row['prediction_id'],**row.to_dict()}, axis=1).tolist()
        
        unique_embeddings = []
        unique_metadata = []

        for row in metadata_entries:
            unique_id = row["_id"]
            if unique_id not in self.added_ids:
                self.added_ids.add(unique_id)

                # if row["chunk1"] not in self.added_chunks:
                #     unique_embeddings.append(row["chunk1_vector"])
                #     unique_metadata.append({row["chunk1"]})
                if row["chunk2"] not in self.added_chunks:
                    unique_embeddings.append(row["chunk2_vector"])
                    unique_metadata.append({"chunk2": row["chunk2"]})
                    self.added_chunks.add(row["chunk2"])
            else:
                print("Found duplicate for row: " + str(row["_id"]))
        
        if unique_embeddings:
            unique_embeddings = np.array(unique_embeddings)
            await self.add(unique_embeddings, unique_metadata)
    
    async def add(self, embeddings: np.ndarray, metadata: List[dict]) -> None:
        #print("Embedding shape before squeezing:", embeddings.shape)
        embeddings = embeddings.squeeze()
        if embeddings.ndim == 1:
        # Reshape if squeezing reduced to 1D
            embeddings = embeddings.reshape(1, -1) # Creates puts all columns into a single row
        #print("Embedding shape after squeezing:", embeddings.shape)
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    async def search(self, query_embedding: np.ndarray, limit: int) -> List[List[Any]]:
        distances, indices = self.index.search(query_embedding, limit)
        batch_results = []
        for distance_row, index_row in zip(distances, indices):
            results = [(self.metadata[idx], dist) for idx, dist in zip(index_row, distance_row)]
            batch_results.append(results[:limit])
        return batch_results

    async def batch_search(self, source_df: DataFrame) -> List[tuple]:
        question_vectors = source_df["chunk1_vector"].tolist()
        batch_vectors = np.vstack(question_vectors)

        distances, indices = self.index.search(batch_vectors, len(self.metadata))
        batch_results = []

        for idx, (distance_row, index_row) in enumerate(zip(distances, indices)):
            source_row = source_df.iloc[idx]
            results = [(self.metadata[idx], dist) for idx, dist in zip(index_row, distance_row)]
            results_sorted = sorted(results, key=lambda x: x[1])  # Sort by distance
            unique_idx = source_row["prediction_id"]
            # if source_row["objective"] == "predict_tail":
            #     tails = [(source_row["head"], source_row["relation"], res[0]['tail'], res[0]['tail_description']) for res in results_sorted]
            #     distinct_tails = list({(head, relation, tail, tail_description) for head, relation, tail, tail_description in tails})
            #top10tails = results_sorted[:10]
            #print(top10tails)

            for result in results_sorted:

                target_chunk = result[0]["chunk2"]
                edge_of_pair = self.get_result(source_row["chunk1"], target_chunk)
                has_edge = edge_of_pair is not None and edge_of_pair != "NONE"

                pred = has_edge

                similarity = 1 / (1 + result[1])  # Convert distance to similarity
                
                batch_results.append((unique_idx, similarity, pred, source_row["chunk1"], target_chunk, has_edge))
                

        return batch_results

    async def clear(self):
        """Just reinitialize to clear."""
        self.index = faiss.IndexHNSWFlat(self.vector_length, self.index_m)
        self.index.hnsw.efConstruction = self.efConstruction
        self.metadata = []
        self.added_ids.clear()
        self.prediction_id_map = {}
        self.added_chunks = set()
        self.result_map = {}

    def get_prediction_id(self, string: str) -> int:
        if string not in self.prediction_id_map:
            self.prediction_id_map[string] = len(self.prediction_id_map)
        return self.prediction_id_map[string]