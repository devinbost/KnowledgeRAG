# Used at eval
# This version uses pretrained model as a base.
import torch
import asyncio
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, Runnable
from langchain.prompts import PromptTemplate
from torchmetrics.retrieval import RetrievalMRR
from torchmetrics.retrieval import RetrievalNormalizedDCG
from torchmetrics.retrieval import RetrievalHitRate
import faiss
import os
import random
import json
import numpy as np
from collections import defaultdict
import re

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# For debugging:
torch.set_printoptions(profile="full")

# If you are using CUDA (GPU), also set this for reproducibility
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
batch_size = 2

# Used at eval
import sentencepiece
print(sentencepiece.__version__)

import pandas as pd
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(api_key=api_key, temperature=0, model_name="gpt-3.5-turbo-0125")

# Run from terminal: conda install -c pytorch -c nvidia faiss-gpu=1.8.0

filename = "knowledge_tuples"
suffix = "T5_knowledge_graph_pretrain_v3_predict_all_torchmetrics"
dataset_prefix = "DataModuleWith_Tail_OneSep"
# Used at eval
save_path = '/teamspace/studios/this_studio/data/'

# These were used for training this file:
special_tokens = [
    "<cls>", "<mask>", "<end>", "<sep>", "<head>", "<tail>", "<relation>"
]
# for i in range(1, 151):
#     special_tokens.append(f"<x{i}>") # These are used for MLM

# special_tokens = [
#     "<cls>", "<head>", "<head_description>", "</head_description>", "<mask>","<relation>", "<tail>", "<tail_description>","</tail_description>","<relation>", "<end>"
# ]
# for i in range(1, 151):
#     special_tokens.append(f"<x{i}>") # These are used for MLM
df_size_limit = 10

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from transformers import T5Config

torch.set_float32_matmul_precision("medium")

vector_length = 512

from pydantic import BaseModel
from typing import Optional, List

class Part(BaseModel):
    text: str
    position: int
    token: str
    bucket: Optional[str] = None
    swapped: bool

class KBDataset(Dataset):
    def __init__(self, tokenizer, df, num_entities, max_length=vector_length):
        self.tokenizer = tokenizer
        self.source_rows = df['Source']
        self.target_rows = df['Target']
        self.max_length = max_length
        self.num_entities = num_entities
        self.df = df

    def __len__(self):
        return len(self.source_rows)

    def __getitem__(self, idx):
        source = self.df['Source'][idx]
        target = self.df['Target'][idx]
        relation = self.df['relation'][idx]
        tail = self.df['tail'][idx]
        tail_description = self.df['tail_description'][idx]
        head = self.df['head'][idx]
        head_description = self.df['head_description'][idx]
        objective = self.df['objective'][idx]

        source_tokenized = self.tokenizer.encode_plus(source, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        target_tokenized = self.tokenizer.encode_plus(target, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        
        source_token_ids = source_tokenized['input_ids'].squeeze()
        attention_mask = source_token_ids.ne(self.tokenizer.pad_token_id)
        target_token_ids = target_tokenized['input_ids'].squeeze()

        return {"source_token_ids": source_token_ids, 
                "attention_mask": attention_mask, 
                "target_token_ids": target_token_ids, 
                "source": source, 
                "target": target, 
                "head": head,
                "head_description": head_description,
                "relation": relation, 
                "tail": tail, 
                "tail_description": tail_description,
                "objective": objective}

class DataModuleWith_Tail_OneSep(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=2, max_length=vector_length, data_file=save_path + filename + '.parquet'):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.data_file = data_file
        self.first_run = True

        self.train_file = os.path.join(save_path, dataset_prefix + 'train_df.parquet')
        self.val_file = os.path.join(save_path, dataset_prefix + 'val_df.parquet')
        self.test_file = os.path.join(save_path, dataset_prefix + 'test_df.parquet')
        self.entities_count_file = os.path.join(save_path, dataset_prefix + 'entities_count.json')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=7)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=7)

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

        # Step 1: List all unique entities
        all_entities = pd.concat([df['head'], df['tail']]).unique()

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
        # train_df = df[df.apply(lambda row: row['head'] in self.train_entities and row['tail'] in self.train_entities, axis=1)].reset_index(drop=True)
        # val_df = df[df.apply(lambda row: row['head'] in self.val_entities and row['tail'] in self.val_entities, axis=1)].reset_index(drop=True)
        # test_df = df[df.apply(lambda row: row['head'] in self.test_entities and row['tail'] in self.test_entities, axis=1)].reset_index(drop=True)

        # This version is for semi-inductive (inductive on head, transductive on tail):
        train_df = df[df.apply(lambda row: row['head'] in self.train_entities, axis=1)].reset_index(drop=True)
        val_df = df[df.apply(lambda row: row['head'] in self.val_entities, axis=1)].reset_index(drop=True)
        test_df = df[df.apply(lambda row: row['head'] in self.test_entities, axis=1)].reset_index(drop=True)


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

    def filter_description_rows(self, row):
        """This method is to prevent leakage of entities in the descriptions to prevent the algorithm from cheating"""
        # Check if 'tail' appears in 'head_description'
        if pd.notna(row['head_description']) and pd.notna(row['tail']) and row['tail'] in row['head_description']:
            return False
        # Check if 'head' appears in 'tail_description'
        if pd.notna(row['tail_description']) and pd.notna(row['head']) and row['head'] in row['tail_description']:
            return False
        # Check if head appears in tail
        if pd.notna(row['tail']) and pd.notna(row['head']) and row['head'] in row['tail']:
            return False

        return True

    def apply_description_filter(self, df):
        # Apply the filter
        print(f"Row count before filter is: {df.shape[0]}")
        mask = df.apply(self.filter_description_rows, axis=1)
        filtered_df = df[mask]
        print(f"Row count after filter is: {filtered_df.shape[0]}")
        return filtered_df

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

                # Filter out cases where information is leaked through the descriptions:
                filtered_df = self.apply_description_filter(df)

                # Split the full dataframe into training, validation, and test sets
                train_df, val_df, test_df = self.create_strict_splits(filtered_df)

                # Apply transformations to each split
                train_df = self.prepare_dataset(train_df)
                val_df = self.prepare_dataset(val_df)
                test_df = self.prepare_dataset(test_df)
                
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
            self.train_dataset = KBDataset(self.tokenizer, train_df, num_train_entities, self.max_length)
            self.val_dataset = KBDataset(self.tokenizer, val_df, num_val_entities, self.max_length)
            self.test_dataset = KBDataset(self.tokenizer, test_df, num_test_entities, self.max_length)
            self.first_run = False

    def sample_parts(self, word_parts: List[Part], percentage: float) -> List[Part]:
        """
        Filter a list of Part objects to only a specified percentage.

        Parameters:
        - parts (List[Part]): The list of Part objects to filter.
        - percentage (float): The percentage of the list to keep.

        Returns:
        - List[Part]: A list containing the specified percentage of Part objects.
        """
        total_count = len(word_parts)
        # Calculate how many parts to select
        selected_count = int(np.ceil(total_count * (percentage / 100)))

        if selected_count < 1:
            # If selected_count is 0, return an empty list or handle as needed
            return []

        # Randomly select indices without replacement
        selected_indices = np.random.choice(total_count, size=selected_count, replace=False)

        # Create a new list with the selected elements
        selected_parts = [word_parts[i] for i in selected_indices]

        return selected_parts

    def generate_word_masks(self, head, head_description, relation, tail, tail_description, percentage):
        combined = f"{head} {head_description} {relation} {tail} {tail_description}"
        head_parts = head.split()
        head_description_parts = head_description.split()
        relation_parts = relation.split()
        tail_parts = tail.split()
        tail_description_parts = tail_description.split()
        combined_text_parts = head_parts + head_description_parts + relation_parts + tail_parts + tail_description_parts
        #assert len(combined_text_parts) == len(combined.split())
        positions = range(0, len(combined_text_parts))
        words = [combined_text_parts[pos] for pos in positions]
        tokens = [f"<x{pos}>" for pos in positions]
        triples = zip(words, positions, tokens)
        combined_parts = [Part(text=word, position=position, token=token, swapped=False) for word, position, token in
                          triples]
        combined_parts_dict = {part.position: part for part in combined_parts}
        # Just for testing:
        # sampled_parts = self.sample_parts(combined_parts, 100)
        # recombined_parts = [word for word, _, _, _, _ in sampled_parts]
        # assert len(recombined_parts) == len(combined_parts)

        sampled_parts = self.sample_parts(combined_parts, percentage)
        for part in sampled_parts:
            combined_parts_dict[part.position] = Part(text=part.token, position=part.position, token=part.text,
                                                      swapped=True)  # Swap text and token for the sampled word
        combined_with_substitution: List[Part] = sorted(combined_parts_dict.values(), key=lambda item: item.position)
        # Split combined_with_substitution into the buckets
        # Function to check range and append to corresponding list
        # Initialize the lists
        head_upper_bound = len(head_parts)
        head_description_upper_bound = len(head_description_parts) + head_upper_bound
        relation_upper_bound = len(relation_parts) + head_description_upper_bound
        tail_upper_bound = len(tail_parts) + relation_upper_bound
        tail_description_upper_bound = len(tail_description_parts) + tail_upper_bound
        head_range = (0, head_upper_bound)
        head_description_range = (head_upper_bound, head_description_upper_bound)
        relation_range = (head_description_upper_bound, relation_upper_bound)
        tail_range = (relation_upper_bound, tail_upper_bound)
        tail_description_range = (tail_upper_bound, tail_description_upper_bound)
        sub_head_parts = []
        sub_head_description_parts = []
        sub_relation_parts = []
        sub_tail_parts = []
        sub_tail_description_parts = []
        for part in combined_with_substitution:
            if head_range[0] <= part.position < head_range[1]:
                part.bucket = "head"
                sub_head_parts.append(part)
            elif head_description_range[0] <= part.position < head_description_range[1]:
                part.bucket = "head_description"
                sub_head_description_parts.append(part)
            elif relation_range[0] <= part.position < relation_range[1]:
                part.bucket = "relation"
                sub_relation_parts.append(part)
            elif tail_range[0] <= part.position < tail_range[1]:
                part.bucket = "tail"
                sub_tail_parts.append(part)
            elif tail_description_range[0] <= part.position < tail_description_range[1]:
                part.bucket = "tail_description"
                sub_tail_description_parts.append(part)
        target_tokens = []
        for part in combined_with_substitution:
            if part.swapped is True:
                target_tokens.append(part.text)
                target_tokens.append(part.token)
        head_with_sub = ' '.join([part.text for part in sub_head_parts])
        head_description_with_sub = ' '.join([part.text for part in sub_head_description_parts])
        relation_with_sub = ' '.join([part.text for part in sub_relation_parts])
        tail_with_sub = ' '.join([part.text for part in sub_tail_parts])
        tail_description_with_sub = ' '.join([part.text for part in sub_tail_description_parts])
        source_text = (f"<cls>Predict missing tokens: Head: {head_with_sub}"
                       f"<head_description>{head_description_with_sub}</head_description>Relation: {relation_with_sub}. Tail: {tail_with_sub}<tail_description>{tail_description_with_sub}</tail_description>")
        target_text = "".join(target_tokens) + "<end>"
        return source_text, target_text

    def prepare_dataset(self, df):
        mem_head = df.apply(lambda x: pd.Series({"Source": f"<cls>predict head: Head: <head><sep>Relation: {x['relation']}<sep>Tail: {x['tail']}<sep>{x['tail_description']}<sep>",
                                                     "Target": f"<head>{x['head']}<end>",
                                                     "relation": x['relation'],
                                                     "tail": x['tail'],
                                                     "tail_description": x['tail_description'],
                                                     "head": x['head'],
                                                     "head_description": x['head_description'],
                                                     "objective": "predict_head"
                                                     }), axis=1)
        mem_tail = df.apply(lambda x: pd.Series({"Source": f"<cls>predict tail: Head: {x['head']}<sep>{x['head_description']}<sep>Relation: {x['relation']}<sep>Tail: <tail><sep>",
                                                     "Target": f"<tail>{x['tail']}<end>",
                                                     "relation": x['relation'],
                                                     "tail": x['tail'],
                                                     "tail_description": x['tail_description'],
                                                     "head": x['head'],
                                                     "head_description": x['head_description'],
                                                     "objective": "predict_tail"
                                                     }), axis=1)

        mem_relation = df.apply(lambda x: pd.Series({"Source": f"<cls>predict relation: Head: {x['head']}<sep>{x['head_description']}<sep>Relation: <relation><sep>Tail: {x['tail']}<sep>{x['tail_description']}<sep>",
                                                     "Target": f"<relation>{x['relation']}<end>",
                                                     "relation": x['relation'],
                                                     "tail": x['tail'],
                                                     "tail_description": x['tail_description'],
                                                     "head": x['head'],
                                                     "head_description": x['head_description'],
                                                     "objective": "predict_relation"
                                                     }), axis=1)

        # Combine both transformations and remove duplicates
        new_df = pd.concat([mem_head, mem_relation, mem_tail]).drop_duplicates().reset_index(drop=True)
        return new_df

# from sentence_transformers import SentenceTransformer
# embedding_model_test = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

class KBModel(pl.LightningModule):
    def __init__(self, data_module, special_tokens, learning_rate=0.001):
        super().__init__()
        self.tokenizer = data_module.tokenizer
        self.train_dataset = data_module.train_dataset
        self.val_dataset = data_module.val_dataset
        self.test_dataset = data_module.test_dataset
        self.learning_rate = learning_rate
        config = T5Config(decoder_start_token_id=self.tokenizer.convert_tokens_to_ids(['<pad>'])[0])
        #self.model = T5ForConditionalGeneration(config)
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.efConstruction = 200
        self.index_m = 32

        self.faiss_index = faiss.IndexHNSWFlat(vector_length, self.index_m)
        self.faiss_index.hnsw.efConstruction = self.efConstruction

        self.ranks_dicts = []

        self.num_predictions = 60 # This is something we may want to vary by config
        self.max_length = 512
        self.max_output_length = 40

        self.metadata = []

        # Variables to track success score and total attempts for accuracy calculation
        self.address_success_score = 0
        self.address_total = 0

        self.index_built = False
        
    def forward(self, source_token_ids, attention_mask=None, target_token_ids=None):
        return self.model(input_ids=source_token_ids, attention_mask=attention_mask, labels=target_token_ids)
    
    def training_step(self, batch, batch_idx):
        # Do I have access to the head, tail, etc here? 
        source_token_ids = batch['source_token_ids']
        attention_mask = batch['attention_mask']
        target_token_ids = batch['target_token_ids']

        # Call the forward method with the required arguments
        outputs = self.forward(source_token_ids=source_token_ids, 
                            attention_mask=attention_mask, 
                            target_token_ids=target_token_ids)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def get_scores(self, ids, scores):
        # ids has row count = self.num_predictions x batch_size
        # scores has tensor for each character. Each tensor is (self.num_predictions x batch_size) in rows, vocab_size in columns
        pad_token_id = self.tokenizer.pad_token_id
        # ids is list of tokenized strings (token sequences)
        # scores is a list of tensors. each tensor contains score of each token in vocab
        # conditioned on ids till that point
        # stack scores
        scores = torch.stack(scores, dim=1) 

        # after stacking, shape is (batch_size*num_return_sequences, num tokens in sequence, vocab size)
        # get probs
        log_probs = torch.log_softmax(scores, dim=2) # Normalize, otherwise we can't sum them
        # remove start token. 
        ids = ids[:, 1:]
        # gather needed probs
        x = ids.unsqueeze(-1).expand(log_probs.shape) # unsqueeze(-1) adds a dimension at the end
        needed_logits = torch.gather(log_probs, 2, x) # Retrieves via indexes, so it gets probabilities that correspond to tokens used
        final_logits = needed_logits[:, :, 0]
        padded_mask = (ids == pad_token_id)
        final_logits[padded_mask] = 0
        final_scores = final_logits.sum(dim=-1) # Sum the probabilities for each token

        return final_scores

    def generate_question_prompt(self) -> PromptTemplate:
        prompt = (
            "You're a text parser. Generate a question when given a source. For example, if your SOURCE is like the following EXAMPLE SOURCE, create a question like EXAMPLE QUESTION."
            "EXAMPLE SOURCE:"
            "Head: iPhone 15. Relation: requires"
            "EXAMPLE QUESTION:"
            "The iPhone 15 requires what?"
            "ACTUAL SOURCE:"
            "Head: {Head}. Relation: {Relation}."
            "ACTUAL QUESTION:"
            )
        return PromptTemplate.from_template(prompt)

    def build_question_creation_chain(self, model: ChatOpenAI) -> Runnable:
        chain = (
            {
                "Head": itemgetter("head"),
                "Relation": itemgetter("relation")
            }
            | self.generate_question_prompt()
            | model
            | StrOutputParser()
            | RunnableLambda(lambda x: "<cls>predict tail:" + x + "<sep>Tail: <tail><sep>") #RunnableLambda(lambda x: "<cls>Predict tail: Head and relation: " + x + " Tail: <tail>")
        )
        return chain

    async def async_invoke(self, question_generation_chain, head, relation):
        # Simulate an async API call
        # This function should be truly asynchronous and handle API calls or async operations
        return await question_generation_chain.ainvoke({"head": head, "relation": relation})
    
    async def generate_topk_questions(self, model: ChatOpenAI, df, limit):
        question_generation_chain = self.build_question_creation_chain(model)
        # Use asyncio to gather results from asynchronous operations
        df_topk = df.head(limit).reset_index(drop=True)
        tasks = []
        for index, row in tqdm(df_topk.iterrows(), total=len(df_topk), desc="Processing row in generate_topk_questions"):
            task = asyncio.create_task(self.async_invoke(question_generation_chain, row['head'], row['relation']))
            tasks.append(task)

        # Await all tasks and get the results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle the exception (log it, retry, or other error handling)
                print("Error encountered:", result)
                results[idx] = np.nan # Ensure the value isn't interpreted as a question

        # Assign results back to the DataFrame
        df['test_question'] = results

        return df

    def prepare_query_vector(self, query_text):
        tokenized_query = self.tokenizer(query_text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to('cuda')
        input_ids = tokenized_query['input_ids'].to('cuda')
        attention_mask = tokenized_query['attention_mask'].to('cuda')

        with torch.no_grad():  # Ensure model is in eval mode and no gradients are computed
            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = encoder_outputs.last_hidden_state
            pooled_embeddings = torch.mean(embeddings, dim=1)  # Pooling over the sequence dimension
        return pooled_embeddings.cpu().numpy()  # Convert tensor to numpy array

    def write_to_faiss(self, batch_size=20):
        with torch.no_grad():
            # Need to clear the index.
            self.faiss_index = faiss.IndexHNSWFlat(vector_length, self.index_m)
            self.faiss_index.hnsw.efConstruction = self.efConstruction
            self.metadata = []

            # Goal is to get validation questions to beat embeddings from training data
            # Combine and deduplicate entries from all datasets
            combined_df = pd.concat([self.train_dataset.df, self.val_dataset.df, self.test_dataset.df]).drop_duplicates().reset_index(drop=True)
            # Filter to only rows where 'objective' is 'predict_tail'
            #filtered_df = combined_df[combined_df['objective'] == "predict_tail"].reset_index(drop=True)
            filtered_df = combined_df
            
            # Calculate the number of batches
            num_batches = len(filtered_df) // batch_size + (0 if len(filtered_df) % batch_size == 0 else 1)

            # Process each batch
            for batch_idx in range(num_batches):
                # Slice the DataFrame to get the batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_df = filtered_df.iloc[start_idx:end_idx]
                
                # Prepare batch data
                sources = batch_df['Source'].tolist()
                tokenized_input = self.tokenizer(sources, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(self.device)
                input_ids = tokenized_input['input_ids'].to(self.device)
                attention_mask = tokenized_input['attention_mask'].to(self.device)
                
                encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = encoder_outputs.last_hidden_state
                
                # Pool embeddings over the sequence dimension
                mean_pooled_output = torch.mean(embeddings, dim=1).detach().cpu().numpy()  # Convert to numpy array for FAISS

                self.faiss_index.add(mean_pooled_output)

                # Store metadata associated with embeddings
                metadata_entries = batch_df.apply(lambda row: (row['objective'], row['head'], row['head_description'],
                                                            row['relation'], row['tail'], row['tail_description']), axis=1).tolist()
                self.metadata.extend(metadata_entries)

            print("FAISS indexing complete and metadata stored.")

    
    def compute_vector_search_mrr(self, dataset):
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
        sources = filtered_df["Source"].tolist()
        question_vectors = [self.prepare_query_vector(source) for source in sources]
        # for vec in question_vectors:
        #     print(f"Individual vector shape: {vec.shape}")  # Check each vector's shape
        batch_vectors = np.stack(question_vectors)
        batch_vectors = batch_vectors.squeeze() # This removes any singleton dimensions

        # print(f"Batch vectors shape: {batch_vectors.shape}")  # Final shape check before FAISS search
        # Search in FAISS in batch
        distances, indices = self.faiss_index.search(batch_vectors, vector_length)

        for source_idx, (distance_row, index_row) in enumerate(zip(distances, indices)):
            if filtered_df.iloc[source_idx]["objective"] == 'predict_tail':
                target = filtered_df.iloc[source_idx]["tail"]
                for distance, idx in zip(distance_row, index_row):
                    full_result = self.metadata[idx] # Use scalar index
                    predicted_result = full_result[4]  

                    eval_result = predicted_result == target
                    # if eval_result:
                    #     print("true")
                    tail_targets.append(eval_result)
                    tail_results.append(1-distance) # Must be passed as similarities to ensure they're sorted correctly.
                    tail_indexes.append(source_idx)

                    all_targets.append(eval_result)
                    all_results.append(1-distance)
                    all_indexes.append(source_idx)
            elif filtered_df.iloc[source_idx]["objective"] == 'predict_head':
                target = filtered_df.iloc[source_idx]["head"]
                for distance, idx in zip(distance_row, index_row):
                    full_result = self.metadata[idx] # Use scalar index
                    predicted_result = full_result[1] 

                    eval_result = predicted_result == target
                    # if eval_result:
                    #     print("true")
                    head_targets.append(eval_result)
                    head_results.append(1-distance)
                    head_indexes.append(source_idx)

                    all_targets.append(eval_result)
                    all_results.append(1-distance)
                    all_indexes.append(source_idx)

            elif filtered_df.iloc[source_idx]["objective"] == 'predict_relation':
                target = filtered_df.iloc[source_idx]["relation"]
                for distance, idx in zip(distance_row, index_row):
                    full_result = self.metadata[idx] # Use scalar index
                    predicted_result = full_result[3] 
                    # if eval_result:
                    #     print("true")

                    eval_result = predicted_result == target
                    relation_targets.append(eval_result)
                    relation_results.append(1-distance)
                    relation_indexes.append(source_idx)

                    all_targets.append(eval_result)
                    all_results.append(1-distance)
                    all_indexes.append(source_idx)
            
        tail_targets_tensor = torch.tensor(tail_targets)
        tail_preds_tensor = torch.tensor(tail_results)
        tail_indexes_tensor = torch.tensor(tail_indexes, dtype=torch.int64)

        head_targets_tensor = torch.tensor(head_targets)
        head_preds_tensor = torch.tensor(head_results)
        head_indexes_tensor = torch.tensor(head_indexes, dtype=torch.int64)

        relation_targets_tensor = torch.tensor(relation_targets)
        relation_preds_tensor = torch.tensor(relation_results)
        relation_indexes_tensor = torch.tensor(relation_indexes, dtype=torch.int64)

        all_targets_tensor = torch.tensor(all_targets)
        all_preds_tensor = torch.tensor(all_results)
        all_indexes_tensor = torch.tensor(all_indexes, dtype=torch.int64)

        def compute_and_log(metric_name, metric_func, preds_tensors, target_tensors, index_tensors, log_func):
            metrics = {}
            for label, preds, targets, indexes in zip(
                ['tail', 'head', 'relation', 'all'], preds_tensors, target_tensors, index_tensors
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

        # Prepare the input tensors in the correct order
        result_tensors = [tail_preds_tensor, head_preds_tensor, relation_preds_tensor, all_preds_tensor]
        target_tensors = [tail_targets_tensor, head_targets_tensor, relation_targets_tensor, all_targets_tensor]
        index_tensors = [tail_indexes_tensor, head_indexes_tensor, relation_indexes_tensor, all_indexes_tensor]

        # Calculate and log the MRRs
        mrrs = compute_and_log("mrr", mrr, result_tensors, target_tensors, index_tensors, self.log)

        mrr1s = compute_and_log("mrr1", mrr1, result_tensors, target_tensors, index_tensors, self.log)
        mrr3s = compute_and_log("mrr3", mrr3, result_tensors, target_tensors, index_tensors, self.log)
        mrr5s = compute_and_log("mrr5", mrr5, result_tensors, target_tensors, index_tensors, self.log)
        mrr10s = compute_and_log("mrr10", mrr10, result_tensors, target_tensors, index_tensors, self.log)

        # Calculate and log the NDCGs
        ndcgs = compute_and_log("ndcg", ndcg, result_tensors, target_tensors, index_tensors, self.log)

        # Calculate and log the Hit Rates
        hr1s = compute_and_log("hr1", hr1, result_tensors, target_tensors, index_tensors, self.log)
        hr3s = compute_and_log("hr3", hr3, result_tensors, target_tensors, index_tensors, self.log)
        hr5s = compute_and_log("hr5", hr5, result_tensors, target_tensors, index_tensors, self.log)
        hr10s = compute_and_log("hr10", hr10, result_tensors, target_tensors, index_tensors, self.log)

        # Print the results
        print(f"Computed MRRs: Tail: {mrrs['tail']}, Head: {mrrs['head']}, Relation: {mrrs['relation']}, All: {mrrs['all']}")
        print(f"Computed MRRs (Top-1): Tail: {mrr1s['tail']}, Head: {mrr1s['head']}, Relation: {mrr1s['relation']}, All: {mrr1s['all']}")
        print(f"Computed MRRs (Top-3): Tail: {mrr3s['tail']}, Head: {mrr3s['head']}, Relation: {mrr3s['relation']}, All: {mrr3s['all']}")
        print(f"Computed MRRs (Top-5): Tail: {mrr5s['tail']}, Head: {mrr5s['head']}, Relation: {mrr5s['relation']}, All: {mrr5s['all']}")
        print(f"Computed MRRs (Top-10): Tail: {mrr10s['tail']}, Head: {mrr10s['head']}, Relation: {mrr10s['relation']}, All: {mrr10s['all']}")

        print(f"Computed NDCGs: Tail: {ndcgs['tail']}, Head: {ndcgs['head']}, Relation: {ndcgs['relation']}, All: {ndcgs['all']}")

        print(f"Computed Hit Rates (Top-1): Tail: {hr1s['tail']}, Head: {hr1s['head']}, Relation: {hr1s['relation']}, All: {hr1s['all']}")
        print(f"Computed Hit Rates (Top-3): Tail: {hr3s['tail']}, Head: {hr3s['head']}, Relation: {hr3s['relation']}, All: {hr3s['all']}")
        print(f"Computed Hit Rates (Top-5): Tail: {hr5s['tail']}, Head: {hr5s['head']}, Relation: {hr5s['relation']}, All: {hr5s['all']}")
        print(f"Computed Hit Rates (Top-10): Tail: {hr10s['tail']}, Head: {hr10s['head']}, Relation: {hr10s['relation']}, All: {hr10s['all']}")

    def clean_text(self, text):
        # if text == "<tail>splash, water and dust resistant</tail>": # Used just for testing
        #     return "splash, water, and dust resistant"
        # Regex pattern to find anything between < and >
        pattern = '<.*?>'
        # Replace the matched text with an empty string
        cleaned_text = re.sub(pattern, '', text)
        stripped = cleaned_text.strip()
        return stripped

    def evaluate(self, batch, mode='val'):
        ranks_dicts = []

        if mode == "val":
            dataset = self.val_dataset
        elif mode == "test":
            dataset = self.test_dataset
        elif mode == "train":
            dataset = self.train_dataset

        ranks_dicts = self.compute_vector_search_mrr(dataset, batch)
        #ranks_dicts = self.compute_model_mrr(batch, dataset, ranks_dicts)

        self.ranks_dicts = ranks_dicts
        return ranks_dicts
    
    def evaluate_all(self):
        dataset = self.test_dataset
        self.compute_vector_search_mrr(dataset)
        

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)


    def validation_step(self, batch, batch_idx):
        source_token_ids = batch['source_token_ids']
        attention_mask = batch['attention_mask']
        target_token_ids = batch['target_token_ids']

        # Call the forward method with the required arguments
        outputs = self.forward(source_token_ids=source_token_ids, 
                            attention_mask=attention_mask, 
                            target_token_ids=target_token_ids)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, logger=True)
        #return self.evaluate(batch, mode='val') 
        return loss 

    def test_step(self, batch, batch_idx):
        source_token_ids = batch['source_token_ids']
        attention_mask = batch['attention_mask']
        target_token_ids = batch['target_token_ids']

        # Call the forward method with the required arguments
        outputs = self.forward(source_token_ids=source_token_ids, 
                            attention_mask=attention_mask, 
                            target_token_ids=target_token_ids)
        loss = outputs.loss
        self.log("test_loss", loss, prog_bar=True, logger=True)

        if not self.index_built:
            self.write_to_faiss()
            self.evaluate_all()
            self.index_built = True

        return loss

def main():
    # Used at eval
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    tokenizer.add_tokens(special_tokens)  
    data_module = DataModuleWith_Tail_OneSep(tokenizer, batch_size=batch_size)
    data_module.setup()
    #model = KBModel(data_module=data_module, special_tokens=special_tokens)
    model = KBModel.load_from_checkpoint("/teamspace/jobs/t5-knowledge-graph-pretrain-v3-predict-all-torchmetrics-2/nodes.0/kg_logs_v4/knowledge_tuples_T5_knowledge_graph_pretrain_v3_predict_all_torchmetrics/version_1/checkpoints/epoch=0-step=1940.ckpt", data_module=data_module, special_tokens=special_tokens)
    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger("kg_logs_v4", name=filename + "_" + suffix)

    
    # Use PyTorch Lightning's Trainer
    trainer = pl.Trainer(max_epochs=50, accelerator="gpu", devices=-1,
                        enable_progress_bar=True,
                        callbacks=[ModelCheckpoint(monitor="val_loss"),
                                    EarlyStopping(monitor="val_loss", patience=3)],
                        logger=logger,
                        precision="16-mixed")

    #trainer.fit(model, datamodule=data_module)
    # model.write_to_faiss()
    # model.evaluate_all()
    trainer.test(model=model, datamodule=data_module)



if __name__ == "__main__":
    main()