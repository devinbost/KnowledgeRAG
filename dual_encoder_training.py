import torch
import asyncio
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, T5Tokenizer, T5EncoderModel, T5Config
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, Runnable
from langchain.prompts import PromptTemplate
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG, RetrievalHitRate
from typing import Any, List, Optional
import faiss
import os
import random
import json
import numpy as np
from collections import defaultdict
import re
from abc import ABC, abstractmethod
from pandas import DataFrame
import pandas as pd
from pydantic import BaseModel

from typing import List, Optional
from pydantic import BaseModel

import sentencepiece
print(sentencepiece.__version__)

import lightning as pl
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# For debugging:
# torch.set_printoptions(profile="full")

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
batch_size = 12

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(api_key=api_key, temperature=0, model_name="gpt-3.5-turbo-0125")

# Run from terminal: conda install -c pytorch -c nvidia faiss-gpu=1.8.0

filename = "coarse_dual_T5"
suffix = "baseline"
dataset_prefix = f"AmexPairs_{suffix}"
# Used at eval

log_name = dataset_prefix + "___" + filename + "__" + suffix
base_path = '/teamspace/studios/this_studio/experiments/dual-encoder/'
save_path = base_path + 'saves'
data_path = base_path + 'data/' + dataset_prefix + '.parquet'

torch.set_float32_matmul_precision("medium")

vector_length = 512
max_length = 1024


class IndexingStrategy(ABC):
    @abstractmethod
    async def add(self, embeddings: np.ndarray, metadata: list) -> None:
        pass

    @abstractmethod
    async def add_batch(self, embeddings_of_batch, dataframe_of_batch) -> None:
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

# Concrete Strategy for FAISS Indexing
class FaissIndexing(IndexingStrategy):
    def __init__(self, vector_length: int, index_m: int, efConstruction: int):
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
            num_batches = len(filtered_df) // batch_size + (0 if len(filtered_df) % batch_size == 0 else 1)

            # Process each batch
            for batch_idx in range(num_batches):
                # Slice the DataFrame to get the batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
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

    def get_prediction_id(self, string: str) -> int:
        if string not in self.prediction_id_map:
            self.prediction_id_map[string] = len(self.prediction_id_map)
        return self.prediction_id_map[string]

# Concrete Strategy for AstraDB Indexing
class AstraDBIndexing(IndexingStrategy):
    def __init__(self, token: str, api_endpoint: str, collection_name: str, vector_length: int, default_limit: int):
        self.astrapy_db = AsyncAstraDB(token=token, api_endpoint=api_endpoint)
        self.collection_name = collection_name
        self.vector_length = vector_length
        self.default_limit = default_limit
        self.collection = None
        self.prediction_id_map = {}

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
            num_batches = len(filtered_df) // batch_size + (0 if len(filtered_df) % batch_size == 0 else 1)

            pair = []

            # Process each batch
            for batch_idx in range(num_batches):
                # Slice the DataFrame to get the batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
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
            # Commented block below is just for debugging/evaluation:
            # Sort the results by $similarity in descending order
            # results_sorted = sorted(results, key=lambda x: x['$similarity'], reverse=True)
            #  # Extract "tail" and "tail_description" when "objective" == "predict_tail"
            # if source_row["objective"] == "predict_tail":
            #     src = source_row["Source"]
            #     tails = [(source_row["head"], source_row["relation"], res['tail'], res['tail_description']) for res in results_sorted]
            #     distinct_tails = list({(head, relation, tail, tail_description) for head, relation, tail, tail_description in tails})
            #     top10tails = distinct_tails[:10]
            #     print(top10tails)

            # if source_row["objective"] == "predict_relation":
            #     # Extract "head" and "tail" when "objective" == "predict_relation"
            #     src = source_row["Source"]
            #     relations = [(source_row['head'], res['relation'], source_row['tail']) for res in results_sorted]
            #     distinct_relations = list({(head, relation, tail) for head, relation, tail in relations})
            #     top10relations = distinct_relations[:10]
            #     print(top10relations)

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

class ContrastiveSentencePairDataset(Dataset):
    def __init__(self, df: DataFrame, tokenizer, max_length=max_length):
        self.sentence_pairs = [
            (row['chunk1'], row['chunk2']) for idx, row in df.iterrows()
        ]
        self.labels = [
            1 if row['result'] != 'NONE' else 0 for idx, row in df.iterrows()
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.df = df

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        chunk1 = self.df['chunk1'][idx]
        chunk2 = self.df['chunk2'][idx]
        result = self.df['result'][idx]

        chunk1_tokenized = self.tokenizer(chunk1, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        chunk2_tokenized = self.tokenizer(chunk2, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        
        chunk1_ids = chunk1_tokenized['input_ids'].squeeze()
        chunk1_attention_mask = chunk1_ids.ne(self.tokenizer.pad_token_id)

        chunk2_ids = chunk2_tokenized['input_ids'].squeeze()
        chunk2_attention_mask = chunk2_ids.ne(self.tokenizer.pad_token_id)

        label = torch.tensor(self.labels[idx], dtype=torch.float)

        return {
            "chunk1": chunk1, 
            "chunk2": chunk2,
            "chunk1_ids": chunk1_ids,
            "chunk2_ids": chunk2_ids,
            "chunk1_attention_mask": chunk1_attention_mask,
            "chunk2_attention_mask": chunk2_attention_mask,
            "result": result,
            "label": label
        }

def collate_fn(batch):
    input_ids1 = torch.cat([item[0]['input_ids'] for item in batch])
    attention_mask1 = torch.cat([item[0]['attention_mask'] for item in batch])
    input_ids2 = torch.cat([item[1]['input_ids'] for item in batch])
    attention_mask2 = torch.cat([item[1]['attention_mask'] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return {'input_ids1': input_ids1, 'attention_mask1': attention_mask1, 'input_ids2': input_ids2, 'attention_mask2': attention_mask2, 'labels': labels}


class ContrastiveDataModule(pl.LightningDataModule):
    def __init__(self, data_path, tokenizer_name='t5-base', batch_size=16, max_length=1024, random_state=42, match_behavior = "omit", train_size=0.98, val_size=0.01, test_size=0.01):
        super().__init__()
        self.data_path = data_path
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_size=train_size
        self.val_size=val_size
        self.test_size=test_size
        assert train_size + val_size + test_size == 1.00
        self.random_state = random_state
        self.tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_name)
        self.first_run = True
        self.data_file = data_path
        self.match_behavior = match_behavior

        full_prefix = f"{dataset_prefix}_{match_behavior}_{train_size}_{val_size}_{test_size}"

        self.train_file = os.path.join(save_path,f"{full_prefix}_train_df.parquet")
        self.val_file = os.path.join(save_path, f"{full_prefix}_val_df.parquet")
        self.test_file = os.path.join(save_path, f"{full_prefix}_test_df.parquet")
        self.entities_count_file = os.path.join(save_path, f"{full_prefix}_entities_count.json")

    def create_strict_splits(self, df, train_size=0.98, val_size=0.01, test_size=0.01):
        """
            Ensure that entities (head and tail) are mutually exclusive between splits. i.e. we don't want leakage between splits
            Note that this approach will result in significant filtering, especially for validation and test splits, but
            that's better than leaking training data into these splits.
            Example usage:
            Assuming df is your DataFrame loaded with 'head' and 'tail' columns.
                train_df, val_df, test_df = create_disjoint_splits(df)
        """
        assert train_size + val_size + test_size == 1, "Split sizes must sum to 1"

        if self.match_behavior == "omit":
            df = df[df['chunk1'] != df['chunk2']]
        elif self.match_behavior == "neg":
            df.loc[df['chunk1'] == df['chunk2'], 'result'] = 'NONE'
        elif self.match_behavior == "pos":
            df.loc[df['chunk1'] == df['chunk2'], 'result'] = 'MATCH'
        else:
            raise Exception("Unknown match parameter provided")
        
        # Sort the dataframe to ensure that the first non-"NONE" result appears first
        df_sorted = df.sort_values(by=['chunk1', 'chunk2', 'result'], ascending=[True, True, True])

        # Group by chunk1 and chunk2 and apply the condition to filter out any extra rows (ensuring the remaining one is not "NONE") 
        def filter_results(group):
            # Check if there are any results that are not "NONE"
            non_none = group[group['result'] != 'NONE']
            if not non_none.empty:
                # If there are, return the first one
                return non_none.iloc[0]
            else:
                # Otherwise, return the first row (which will have "NONE" as the result)
                return group.iloc[0]

        filtered_df = df_sorted.groupby(['chunk1', 'chunk2']).apply(filter_results).reset_index(drop=True)
        # Step 1: List all unique chunks
        all_entities = pd.concat([filtered_df['chunk1'], filtered_df['chunk2']]).unique()

        # Step 2: Shuffle entities
        np.random.seed(42)
        np.random.shuffle(all_entities)

        # Step 3: Allocate entities to splits based on defined sizes
        total_entities = len(all_entities)
        train_end = int(np.floor(train_size * total_entities))
        val_end = train_end + int(np.floor(val_size * total_entities))
        
        self.train_entities = set(all_entities[:train_end])
        self.val_entities = set(all_entities[train_end:val_end])
        self.test_entities = set(all_entities[val_end:])

        # Step 4: Assign rows to splits based on entity allocation
        # This version is for inductive on both head and tail:
        train_df = filtered_df[filtered_df.apply(lambda row: row['chunk1'] in self.train_entities and row['chunk2'] in self.train_entities, axis=1)].reset_index(drop=True)
        val_df = filtered_df[filtered_df.apply(lambda row: row['chunk1'] in self.val_entities and row['chunk2'] in self.val_entities, axis=1)].reset_index(drop=True)
        test_df = filtered_df[filtered_df.apply(lambda row: row['chunk1'] in self.test_entities and row['chunk2'] in self.test_entities, axis=1)].reset_index(drop=True)

        # # This version is for semi-inductive (inductive on head, transductive on tail):
        # train_df = df[df.apply(lambda row: row['head'] in self.train_entities, axis=1)].reset_index(drop=True)
        # val_df = df[df.apply(lambda row: row['head'] in self.val_entities, axis=1)].reset_index(drop=True)
        # test_df = df[df.apply(lambda row: row['head'] in self.test_entities, axis=1)].reset_index(drop=True)


        print(f"Row count of train_df is: {train_df.shape[0]}")
        print(f"Row count of val_df is: {val_df.shape[0]}")
        print(f"Row count of test_df is: {test_df.shape[0]}")

        # print("Train:")
        # print(train_df[['head','relation','tail']].head())
        # print("Val:")
        # print(val_df[['head','relation','tail']].head())
        # print("Test:")
        # print(test_df[['head','head_description','relation','tail', 'tail_description']].head())

        return train_df, val_df, test_df

    def setup(self, stage=None):
        if self.first_run:
            if os.path.exists(self.train_file) and os.path.exists(self.val_file) and os.path.exists(self.test_file) and os.path.exists(self.entities_count_file):
                # Load the datasets if they already exist
                print("Loading preprocessed datasets...")
                train_df = pd.read_parquet(self.train_file)
                val_df = pd.read_parquet(self.val_file)
                test_df = pd.read_parquet(self.test_file)
                with open(self.entities_count_file, 'r') as f:
                    entities_count = json.load(f)
                num_train_entities = entities_count['num_train_entities']
                num_val_entities = entities_count['num_val_entities']
                num_test_entities = entities_count['num_test_entities']
            else:
            # Load the dataset
                df = pd.read_parquet(self.data_file)

                # # Filter out cases where information is leaked through the descriptions:
                # filtered_df = self.apply_description_filter(df)

                # Split the full dataframe into training, validation, and test sets
                train_df, val_df, test_df = self.create_strict_splits(df, 
                    train_size=self.train_size, 
                    val_size=self.val_size, 
                    test_size=self.test_size)

                # # Apply transformations to each split
                # train_df = self.prepare_dataset(train_df)
                # val_df = self.prepare_dataset(val_df)
                # test_df = self.prepare_dataset(test_df)
                
                train_df.to_parquet(self.train_file)
                val_df.to_parquet(self.val_file)
                test_df.to_parquet(self.test_file)

                num_train_entities = len(self.train_entities)
                num_val_entities = len(self.val_entities)
                num_test_entities = len(self.test_entities)
                entities_count = {
                    'num_train_entities': num_train_entities,
                    'num_val_entities': num_val_entities,
                    'num_test_entities': num_test_entities
                }
                with open(self.entities_count_file, 'w') as f:
                    json.dump(entities_count, f)

                print(f"Row count of train_df is: {train_df.shape[0]}")
                print(f"Row count of val_df is: {val_df.shape[0]}")
                print(f"Row count of test_df is: {test_df.shape[0]}")
            # Assign dataset instances for each split
            self.train_dataset = ContrastiveSentencePairDataset(train_df, self.tokenizer, self.max_length)
            self.val_dataset = ContrastiveSentencePairDataset(val_df, self.tokenizer, self.max_length)
            self.test_dataset = ContrastiveSentencePairDataset(test_df, self.tokenizer, self.max_length)
            self.first_run = False

    
    def train_dataloader(self):
        print(f"Train DataLoader batch size: {self.batch_size}")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)

    def val_dataloader(self):
        print(f"Validation DataLoader batch size: {self.batch_size}")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=7)

    def test_dataloader(self):
        print(f"Test DataLoader batch size: {self.batch_size}")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=7)

class DualEncoderT5Contrastive(pl.LightningModule):
    def __init__(self, data_module: ContrastiveDataModule, model_name: str, indexing_strategy: IndexingStrategy, learning_rate=1e-3, temperature=0.05):
        # super(DualEncoderT5Contrastive, self).__init__()
        super().__init__()
        self.data_module = data_module
        self.tokenizer = data_module.tokenizer
        self.train_dataset = data_module.train_dataset
        self.val_dataset = data_module.val_dataset
        self.test_dataset = data_module.test_dataset

        self.encoder1 = T5EncoderModel.from_pretrained(model_name)
        self.encoder2 = T5EncoderModel.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.temperature = temperature

        self.index_built = False

        self.indexing_strategy = indexing_strategy

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def contrastive_loss(self, z1, z2, labels):
        logits = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0), device=self.device)
        return F.cross_entropy(logits, labels)

    def forward(self, chunk1_ids, chunk1_attention_mask, chunk2_ids, chunk2_attention_mask):
        encoder1_outputs = self.encoder1(input_ids=chunk1_ids, attention_mask=chunk1_attention_mask)
        encoder2_outputs = self.encoder2(input_ids=chunk2_ids, attention_mask=chunk2_attention_mask)
        
        pooled_output1 = self.mean_pooling(encoder1_outputs, chunk1_attention_mask)
        pooled_output2 = self.mean_pooling(encoder2_outputs, chunk2_attention_mask)
        
        normalized_output1 = F.normalize(pooled_output1, p=2, dim=1)
        normalized_output2 = F.normalize(pooled_output2, p=2, dim=1)
        
        return normalized_output1, normalized_output2

    def compute_step(self, batch, batch_idx):
        chunk1_ids = batch["chunk1_ids"]
        chunk2_ids = batch["chunk2_ids"]
        chunk1_attention_mask = batch["chunk1_attention_mask"]
        chunk2_attention_mask = batch["chunk2_attention_mask"]
        
        labels = batch['label']
        z1, z2 = self.forward(chunk1_ids, chunk1_attention_mask, chunk2_ids, chunk2_attention_mask)
        loss = self.contrastive_loss(z1, z2, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, logger=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, logger=True, batch_size=len(batch))
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True, logger=True, batch_size=len(batch))
        asyncio.run(self.index_and_evaluate())
        return loss

    async def index_and_evaluate(self):
        if not self.index_built:
            await self.indexing_strategy.clear()
            #await self.write_to_index()
            await self.evaluate_all()
            self.index_built = True
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def prepare_query_vector(self, chunk_text:str, encoder: int):
        tokenized_chunk = self.tokenizer(chunk_text, return_tensors='pt', padding=True, truncation=True, max_length=self.data_module.max_length).to(self.device)
        chunk_ids = tokenized_chunk['input_ids'].to(self.device)
        chunk_attention_mask = tokenized_chunk['attention_mask'].to(self.device)

        with torch.no_grad():
            if encoder == 1:
                encoder_outputs = self.encoder1(input_ids=chunk_ids, attention_mask=chunk_attention_mask)
            elif encoder == 2:
                encoder_outputs = self.encoder2(input_ids=chunk_ids, attention_mask=chunk_attention_mask)
            else:
                raise

            pooled_output = self.mean_pooling(encoder_outputs, chunk_attention_mask)
            normalized_output = F.normalize(pooled_output, p=2, dim=1)

        return normalized_output.cpu().numpy()  # Convert tensor to numpy array

    async def compute_vector_search_mrr(self, dataset):
        # Index into vector search

        # Get top items randomly from DF
        #  filtered_df = dataset.df[ (dataset.df['objective'] == 'predict_tail')].sample(frac=1, random_state=seed).head(df_size_limit).reset_index(drop=True)
        filtered_df = dataset.df
        #filtered_df = dataset.df.sample(frac=1, random_state=seed).head(df_size_limit).reset_index(drop=True)
        print(f"ANN df is length: {len(filtered_df)}")

        tail_targets = []
        tail_results = []
        tail_indexes = []

        head_targets = []
        head_results = []
        head_indexes = []
        
        relation_targets = []
        relation_results = []
        relation_indexes = []

        all_targets = []
        all_results = []
        all_indexes = []

        # Prepare data for vectorized operations
        #sources = filtered_df["Source"].tolist()
        
        filtered_df['chunk1_vector'] = filtered_df.apply(lambda row: self.prepare_query_vector(row['chunk1'], encoder=1), axis=1)
        filtered_df['chunk2_vector'] = filtered_df.apply(lambda row: self.prepare_query_vector(row['chunk2'], encoder=2), axis=1)
        filtered_df['chunk1_id'] = filtered_df.apply(lambda row: self.indexing_strategy.get_prediction_id(row['chunk1']), axis=1)
        filtered_df['chunk2_id'] = filtered_df.apply(lambda row: self.indexing_strategy.get_prediction_id(row['chunk2']), axis=1)
        filtered_df['prediction_id'] = filtered_df.apply(lambda row: self.indexing_strategy.get_prediction_id(f"{row['chunk1']}|{row['chunk2']}"), axis=1)

        await self.indexing_strategy.add_all(filtered_df)
        #question_vectors = [self.prepare_query_vector(source) for source in sources]
        # for vec in question_vectors:
        #     print(f"Individual vector shape: {vec.shape}")  # Check each vector's shape
        
        all_result_list = await self.indexing_strategy.batch_search(filtered_df) # (unique_idx, similarity/pred, true/false, objective)

        # tail_list = [r for r in all_result_list if r[3] == "predict_tail"]
        # head_list = [r for r in all_result_list if r[3] == "predict_head"]
        # relation_list = [r for r in all_result_list if r[3] == "predict_relation"]

        # Unpack and convert to lists
        all_unique_idx_list, all_similarity_list, all_true_false_list, all_chunk1_list, all_chunk2_list, all_has_edge_list = zip(*all_result_list)
        
        all_unique_idx_list = list(all_unique_idx_list)
        all_similarity_list = list(all_similarity_list)
        all_true_false_list = list(all_true_false_list)

        # Convert to tensors
        all_indexes_tensor = torch.tensor(all_unique_idx_list, dtype=torch.int64)
        all_preds_tensor = torch.tensor(all_similarity_list)
        all_targets_tensor = torch.tensor(all_true_false_list)
        

        def compute_and_log(metric_name, metric_func, preds_tensors, target_tensors, index_tensors, log_func):
            metrics = {}
            for label, preds, targets, indexes in zip(
                ['all'], preds_tensors, target_tensors, index_tensors
            ):
                metric_result = metric_func(preds, targets, indexes=indexes)
                metrics[label] = metric_result.item()
                log_func(f"{metric_name}_predict_{label}", metrics[label], prog_bar=True, logger=True)
            return metrics

        # Initialize metric functions
        mrr = RetrievalMRR()
        mrr1 = RetrievalMRR(top_k=1)
        mrr3 = RetrievalMRR(top_k=3)
        mrr5 = RetrievalMRR(top_k=5)
        mrr10 = RetrievalMRR(top_k=10)

        ndcg = RetrievalNormalizedDCG()

        hr1 = RetrievalHitRate(top_k=1)
        hr3 = RetrievalHitRate(top_k=3)
        hr5 = RetrievalHitRate(top_k=5)
        hr10 = RetrievalHitRate(top_k=10)

        # Add other tensors to lists if more exist for the use case:
        result_tensors = [all_preds_tensor]
        target_tensors = [all_targets_tensor]
        index_tensors = [all_indexes_tensor]

        # Calculate and log the MRRs
        mrrs = compute_and_log("mrr", mrr, result_tensors, target_tensors, index_tensors, self.log)

        mrr1s = compute_and_log("mrr01", mrr1, result_tensors, target_tensors, index_tensors, self.log)
        mrr3s = compute_and_log("mrr03", mrr3, result_tensors, target_tensors, index_tensors, self.log)
        mrr5s = compute_and_log("mrr05", mrr5, result_tensors, target_tensors, index_tensors, self.log)
        mrr10s = compute_and_log("mrr10", mrr10, result_tensors, target_tensors, index_tensors, self.log)

        # Calculate and log the NDCGs
        ndcgs = compute_and_log("ndcg", ndcg, result_tensors, target_tensors, index_tensors, self.log)

        # Calculate and log the Hit Rates
        hr1s = compute_and_log("hr01", hr1, result_tensors, target_tensors, index_tensors, self.log)
        hr3s = compute_and_log("hr03", hr3, result_tensors, target_tensors, index_tensors, self.log)
        hr5s = compute_and_log("hr05", hr5, result_tensors, target_tensors, index_tensors, self.log)
        hr10s = compute_and_log("hr10", hr10, result_tensors, target_tensors, index_tensors, self.log)

        # Print the results
        print(f"Computed MRRs: All: {mrrs['all']}")
        print(f"Computed MRRs (Top-1): All: {mrr1s['all']}")
        print(f"Computed MRRs (Top-3): All: {mrr3s['all']}")
        print(f"Computed MRRs (Top-5): All: {mrr5s['all']}")
        print(f"Computed MRRs (Top-10): All: {mrr10s['all']}")

        print(f"Computed NDCGs: All: {ndcgs['all']}")

        print(f"Computed Hit Rates (Top-1): All: {hr1s['all']}")
        print(f"Computed Hit Rates (Top-3): All: {hr3s['all']}")
        print(f"Computed Hit Rates (Top-5): All: {hr5s['all']}")
        print(f"Computed Hit Rates (Top-10): All: {hr10s['all']}")

    async def evaluate(self, batch, mode='val'):
        ranks_dicts = []

        if mode == "val":
            dataset = self.val_dataset
        elif mode == "test":
            dataset = self.test_dataset
        elif mode == "train":
            dataset = self.train_dataset

        ranks_dicts = await self.compute_vector_search_mrr(dataset)
        #ranks_dicts = self.compute_model_mrr(batch, dataset, ranks_dicts)

        self.ranks_dicts = ranks_dicts
        return ranks_dicts
    
    async def evaluate_all(self):
        dataset = self.test_dataset
        await self.compute_vector_search_mrr(dataset)


def main():
   
    # Initialize
    data_module = ContrastiveDataModule(data_path=data_path, 
        tokenizer_name='t5-small', 
        batch_size=16, 
        max_length=1024, 
        match_behavior = "neg",
        train_size=0.98, 
        val_size=0.0, 
        test_size=0.02)
    data_module.setup()
    faiss_indexing_strategy = FaissIndexing(vector_length=vector_length, index_m=32, efConstruction=200)
    # astradb_indexing_strategy = AstraDBIndexing(
    #     token=os.getenv("ASTRA_TOKEN"), 
    #     api_endpoint=os.getenv("ASTRA_API_ENDPOINT"), 
    #     collection_name=os.getenv("ASTRA_COLLECTION"), 
    #     vector_length=vector_length, 
    #     default_limit=10
    # )

    model = DualEncoderT5Contrastive(data_module=data_module, 
        model_name='t5-small',
        indexing_strategy=faiss_indexing_strategy, 
        learning_rate=1e-3, 
        temperature=0.05)
    # Initialize trainer

    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger(filename, name=log_name)

    trainer = pl.Trainer(max_epochs=50, accelerator="gpu", devices=-1,
                        enable_progress_bar=True,
                        callbacks=[ModelCheckpoint(monitor="val_loss"),
                                    EarlyStopping(monitor="val_loss", patience=3)],
                        logger=logger,
                        precision="16-mixed")

    

    # Train the model
    #trainer.fit(model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)



if __name__ == "__main__":
    main()
