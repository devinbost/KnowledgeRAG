from dual_encoder.IndexingStrategy import IndexingStrategy
from astrapy import AsyncAstraDB, AsyncAstraDBCollection
import asyncio
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
import torch

# Concrete Strategy for AstraDB Indexing

seed = 42
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

#TODO: Needs to be updated for the dual encoder approach
class AstraDBIndexing(IndexingStrategy):
    def __init__(self, token: str, api_endpoint: str, collection_name: str, vector_length: int, default_limit: int, batch_size: int = 16):
        self.astrapy_db = AsyncAstraDB(token=token, api_endpoint=api_endpoint)
        self.collection_name = collection_name
        self.vector_length = vector_length
        self.default_limit = default_limit
        self.collection = None
        self.prediction_id_map = {}
        self.batch_size = batch_size

    async def initialize_collection(self):
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        mycollections = await self.astrapy_db.get_collections()
        if self.collection_name not in mycollections["status"]["collections"]:
            self.collection = await self.astrapy_db.create_collection(collection_name=self.collection_name, dimension=self.vector_length)
        else:
            self.collection = AsyncAstraDBCollection(collection_name=self.collection_name, astra_db=self.astrapy_db)

    async def add_all(self, full_dataframe: DataFrame):
        with torch.no_grad():
            filtered_df = full_dataframe
            
            # Calculate the number of batches
            num_batches = len(filtered_df) // self.batch_size + (0 if len(filtered_df) % self.batch_size == 0 else 1)

            pair = []

            # Process each batch
            for batch_idx in range(num_batches):
                # Slice the DataFrame to get the batch
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_df = filtered_df.iloc[start_idx:end_idx]

                await self.add_batch(batch_df)
    
    async def add_batch(self, dataframe_of_batch) -> None:
        ready_to_write = []
        
        for index, row in dataframe_of_batch.iterrows():
            if isinstance(row["SourceVector"], np.ndarray):
                row["SourceVector"] = row["SourceVector"].squeeze().tolist()
            unique_id = row['prediction_id']
            prediction_id = self.get_prediction_id(unique_id)
            row["prediction_id"] = prediction_id
            new_obj = {"$vector": row["SourceVector"], "_id": unique_id, **row}
            ready_to_write.append(new_obj)
        
        await self.write_batch(ready_to_write)

    async def write_batch(self, batch) -> None:
        if self.collection is None:
            await self.initialize_collection()
        try:
            await self.collection.insert_many(batch, partial_failures_allowed=True)
        except Exception as e:
            print(f"Error: {e}")
    
    async def add(self, embeddings: np.ndarray, metadata: list) -> None:
        for embed, meta in tqdm(zip(embeddings, metadata), total=len(metadata)):
            document = {"_id": meta[6]["Source"], "$vector": embed.tolist(), **meta[6]} # Caution: meta[6] currently is the dict, but be careful the contract doesn't change.
            try:
                await self.collection.insert_one(document)
            except Exception as e:
                print(f"Error: {e}")
    
    async def search(self, query_embedding: np.ndarray, limit: int) -> list:
        await self.initialize_collection()
        result = await self.collection.vector_find(
            vector=query_embedding.tolist(),
            limit=limit
        )
        return result
    
    async def batch_search(self, source_df: DataFrame):
        batch_results = []
        for idx, source_row in source_df.iterrows():
            results = await self.collection.vector_find(
                vector=source_row["SourceVector"].squeeze().tolist(),
                limit=self.default_limit
            )

            for result in results:
                unique_idx = source_row["prediction_id"]
                similarity = result['$similarity']

                if source_row["objective"] == "predict_tail":
                    pred = source_row["tail"] == result["tail"]
                    if pred == True:
                        print(source_row["Source"])
                    batch_results.append((unique_idx, similarity, pred, "predict_tail"))
                elif source_row["objective"] == "predict_head":
                    pred = source_row["head"] == result["head"]
                    batch_results.append((unique_idx, similarity, pred, "predict_head"))
                elif source_row["objective"] == "predict_relation":
                    pred = source_row["relation"] == result["relation"]
                    batch_results.append((unique_idx, similarity, pred, "predict_relation"))
        return batch_results

    async def clear(self):
        await self.initialize_collection()
        await self.collection.clear()

    def get_prediction_id(self, string):
        if string not in self.prediction_id_map:
            self.prediction_id_map[string] = len(self.prediction_id_map)
        return self.prediction_id_map[string]