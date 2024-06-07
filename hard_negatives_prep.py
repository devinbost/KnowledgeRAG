import os
from dotenv import load_dotenv
import asyncio
import random
import re
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from cassandra.concurrent import execute_concurrent_with_args
from networkx import DiGraph
from pandas import DataFrame
from tqdm import tqdm
from transformers import T5Tokenizer
from pydantic import BaseModel
from pydantic.warnings import PydanticDeprecatedSince20
from astrapy.db import AsyncAstraDB, AsyncAstraDBCollection

seed = 42
random.seed(seed)
np.random.seed(seed)

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, Runnable
import json
import logging
from operator import itemgetter
import getpass
from sentence_transformers import SentenceTransformer

# Load .env file
load_dotenv()

class DataProcessor:
    def build_coarse_embeddings(self, df_with_embeddings: pd.DataFrame, batch_size: int = 4096):
        embedding_model = SentenceTransformer('thenlper/gte-small')
        num_rows = len(df_with_embeddings)
        
        chunk1_embeddings = []
        chunk2_embeddings = []
        
        for start_idx in tqdm(range(0, num_rows, batch_size), desc="Encoding Batches"):
            end_idx = min(start_idx + batch_size, num_rows)
            chunk1_batch = df_with_embeddings['chunk1'][start_idx:end_idx].tolist()
            chunk2_batch = df_with_embeddings['chunk2'][start_idx:end_idx].tolist()
            
            chunk1_embeddings.extend(embedding_model.encode(chunk1_batch))
            chunk2_embeddings.extend(embedding_model.encode(chunk2_batch))
        
        df_with_embeddings['chunk1_embedding'] = chunk1_embeddings
        df_with_embeddings['chunk2_embedding'] = chunk2_embeddings

    def get_related_chunks(self, G: DiGraph, chunk1: str):
        successors = set(G.successors(chunk1))
        related_nodes = {node for node in successors if G[chunk1][node]['relation'] != "NONE"}
        return list(related_nodes)

    from tqdm.asyncio import tqdm
    async def find_negatives(self, row, G, chunk_embedding_collection, include_self_as_negative: bool, initial_query_limit: int, negative_limit: int):
        related_chunks = self.get_related_chunks(G, row["chunk1"])
        related_chunks_trimmed = related_chunks[:100]
        negatives = set()
        limit = initial_query_limit
        
        while len(negatives) < negative_limit:
            known_negatives = await chunk_embedding_collection.vector_find(
                filter={"chunk2": {"$nin": related_chunks_trimmed}},
                vector=row["chunk1_embedding"].tolist(),
                limit=limit
            )
            for result in known_negatives:
                if result not in related_chunks:
                    negatives.add(result["chunk2"])
            limit *= 2  # Increase the limit exponentially
        
        negatives_list = list(negatives)[:negative_limit]
        if include_self_as_negative:
            if row["chunk1"] not in negatives_list:
                if len(negatives_list) < negative_limit:
                    negatives_list.append(row["chunk1"])  # Including self in negatives
                else:
                    negatives_list[-1] = row["chunk1"]  # Replace the last item to maintain the limit

        return negatives_list

    async def add_negatives_to_dataframe(self, df: DataFrame, G: DiGraph, chunk_embedding_collection, initial_query_limit: int, include_self_as_negative: bool, negative_limit: int, batch_size=100):
        num_rows = len(df)
        similar_negatives = []
        
        for start_idx in tqdm(range(0, num_rows, batch_size), desc="Finding Negatives"):
            end_idx = min(start_idx + batch_size, num_rows)
            batch_df = df.iloc[start_idx:end_idx]
            tasks = [self.find_negatives(row, G, chunk_embedding_collection, include_self_as_negative, initial_query_limit, negative_limit) for _, row in batch_df.iterrows()]
            batch_negatives = await asyncio.gather(*tasks)
            similar_negatives.extend(batch_negatives)
        
        df['similar_negatives'] = similar_negatives
        return df

    async def build_dataframe_with_negatives(self, input_file: str, suffix: str, test: bool, initial_query_limit: int, include_self_as_negative: bool, negative_limit: int):
        # Read the parquet file into a DataFrame
        df_with_embeddings = pd.read_parquet(input_file)
        #df_with_embeddings = df_with_embeddings[df_with_embeddings["chunk1"] != df_with_embeddings["chunk2"]]
        if test:
            df_with_embeddings["chunk1"] = df_with_embeddings["chunk1"].apply(lambda x: x[:20])
            df_with_embeddings["chunk2"] = df_with_embeddings["chunk2"].apply(lambda x: x[:20])
        
        df_with_embeddings = df_with_embeddings[~df_with_embeddings['result'].str.contains("NONE", na=False)]
        #df_with_embeddings = df_with_embeddings[df_with_embeddings['result'] != 'NONE']

        def filter_df(df):
            # Group by chunk1
            grouped = df.groupby(['chunk1', 'chunk2'])

            # Function to filter each group
            def filter_group(group):
                if len(group) == 1:
                    return group
                # Check if at least one row has result != "NONE"
                if (group['result'] != 'NONE').sum() > 0:
                    # Filter out rows where result == "NONE"
                    return group[group['result'] != 'NONE']
                else:
                    # Return the group as is
                    return group

            # Apply the filter function to each group
            filtered_groups = grouped.apply(filter_group)

            # Reset index to get the final DataFrame
            filtered_df = filtered_groups.reset_index(drop=True)

            return filtered_df

        df_with_embeddings = filter_df(df_with_embeddings)
        G = nx.DiGraph()

        df_with_embeddings.apply(lambda row: G.add_edge(row["chunk1"], row["chunk2"], relation=row["result"]), axis=1)
        self.build_coarse_embeddings(df_with_embeddings)

        astrapy_db = AsyncAstraDB(token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
                                  api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"])
        await astrapy_db.delete_collection("pairs_for_embeddings_small_dual_encoder")
        chunk_embedding_collection = await astrapy_db.create_collection(
            collection_name="pairs_for_embeddings_small_dual_encoder", dimension=384)  # Embed just the chunk content

        await self.write_vectors(df_with_embeddings, chunk_embedding_collection)

        df_with_negatives = await self.add_negatives_to_dataframe(df_with_embeddings, G, chunk_embedding_collection, initial_query_limit, include_self_as_negative, negative_limit)

        result_file_name = f"pairs_for_embeddings_with_negatives{suffix}.parquet"
        return df_with_negatives

    async def write_vectors(self, df_with_embeddings: DataFrame, chunk_embedding_collection: AsyncAstraDBCollection):
        # Compute huggingface embedding for chunk1 and chunk2 via df_with_embeddings.apply(.., dim=1)
        # Use async batched write to write all rows to Astra after setting chunk2_embedding to "$vector"
        # Then, use async find for each row to get top 20
        #

        batch_size = 20
        import numpy as np
        # Calculate the number of batches
        num_batches = len(df_with_embeddings) // batch_size + (0 if len(df_with_embeddings) % batch_size == 0 else 1)
        pair = []
        with tqdm(total=num_batches, desc="Processing Batches") as pbar:
            # Process each batch
            for batch_idx in range(num_batches):
                # Slice the DataFrame to get the batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_df = df_with_embeddings.iloc[start_idx:end_idx]

                ready_to_write_chunk = []

                for index, row in batch_df.iterrows():
                    if isinstance(row["chunk1_embedding"], np.ndarray):
                        row["chunk1_embedding"] = row["chunk1_embedding"].squeeze().tolist()

                    chunk1_obj = {"$vector": row["chunk2_embedding"], "chunk1": row["chunk1"], "chunk2": row["chunk2"],
                                  "result": str(row["result"])}

                    ready_to_write_chunk.append(chunk1_obj)
                await chunk_embedding_collection.insert_many(ready_to_write_chunk, partial_failures_allowed=True)
                pbar.update(1)

def main():
    processor = DataProcessor()
    suffix = "_full"
    input_file_name = f"pairs_for_embeddings{suffix}.parquet"
    output_file_name = f"pairs_for_embeddings{suffix}_out_directed_top20neg.parquet"
    directory = "/teamspace/studios/this_studio/experiments/data_prep/"
    input_file = directory + input_file_name
    output_file = directory + output_file_name
    df_with_negatives = asyncio.run(processor.build_dataframe_with_negatives(input_file, suffix, test=False, initial_query_limit=20, include_self_as_negative=True, negative_limit=20))
    df_with_negatives = df_with_negatives[df_with_negatives["chunk1"] != df_with_negatives["chunk2"]]
    #updated_df["result"] = updated_df["result"].apply(lambda x: "NONE")
    #filtered_df = updated_df[~updated_df['result'].str.contains("NONE", na=False)]
    

    assert len(df_with_negatives) > 0, "Assertion failed. filtered_df was empty"

    for index, row in df_with_negatives.iterrows():
        #print(f"Row chunk1 is: {row['chunk1'][:20]}")
        #print(f"Row chunk2 is: {row['chunk2'][:20]}")
        for item in row["similar_negatives"]:
            print(f"Negative is: {item[:20]}")
        assert row["chunk2"] not in row["similar_negatives"], f"Assertion failed at index {index}: chunk2 is in similar_negatives"
        assert row["chunk1"] in row["similar_negatives"], f"Assertion failed at index {index}: chunk1 is not in similar_negatives"
        #print(f"list len: {len(row['similar_negatives'])}; set len: {len(set(row['similar_negatives']))}")
        assert len(row["similar_negatives"]) == len(set(row["similar_negatives"])), f"Assertion failed at index {index}: Lengths do not match"
        assert len(row["similar_negatives"]) == 20, f"Assertion failed at index {index}: similar_negatives was not 20 in length"
    print("All assertions passed.")
    
    df_with_negatives.to_parquet(output_file)

if __name__ == "__main__":
    main()