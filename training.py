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

# If you are using CUDA (GPU), also set this for reproducibility
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
batch_size = 12

# Used at eval
import sentencepiece
print(sentencepiece.__version__)

import pandas as pd

# Run from terminal: conda install -c pytorch -c nvidia faiss-gpu=1.8.0

filename = "knowledge_tuples"
suffix = "T5_knowledge_graph_pretrain_v3_predict_tail_no_mlm"
dataset_prefix = "DataModuleWith_Tail_NoMLM"
# Used at eval
save_path = '/teamspace/studios/this_studio/data/'

special_tokens = [
    "<cls>", "<head>", "<head_description>", "</head_description>", "<mask>","<relation>", "<tail>", "<tail_description>","</tail_description>","<relation>", "<end>", "<context>", "</context>"
]
# for i in range(1, 151):
#     special_tokens.append(f"<x{i}>") # These are used for MLM

# Used at eval

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from transformers import T5Config

torch.set_float32_matmul_precision("medium")

vector_length = 1024

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
        self.source_rows = df['source']
        self.target_rows = df['target']
        self.max_length = max_length
        self.num_entities = num_entities
        self.df = df

    def __len__(self):
        return len(self.source_rows)

    def __getitem__(self, idx):
        source = self.df['source'][idx]
        target = self.df['target'][idx]
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

class DataModuleWith_Tail_NoMLM(pl.LightningDataModule):
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
        mem_head = df.apply(lambda x: pd.Series({"source": f"<cls>predict head: Head: <head>. Relation: {x['relation']}. Tail: {x['tail']}<tail_description>{x['tail_description']}</tail_description>",
                                                     "target": f"<head>{x['head']}<end>",
                                                     "relation": x['relation'],
                                                     "tail": x['tail'],
                                                     "tail_description": x['tail_description'],
                                                     "head": x['head'],
                                                     "head_description": x['head_description'],
                                                     "objective": "predict_head"
                                                     }), axis=1)
        mem_tail = df.apply(lambda x: pd.Series({"source": f"<cls>predict tail: Head: {x['head']}<head_description>{x['head_description']}</head_description>Relation: {x['relation']}. Tail: <tail>",
                                                     "target": f"<tail>{x['tail']}<end>",
                                                     "relation": x['relation'],
                                                     "tail": x['tail'],
                                                     "tail_description": x['tail_description'],
                                                     "head": x['head'],
                                                     "head_description": x['head_description'],
                                                     "objective": "predict_tail"
                                                     }), axis=1)

        # mem_tail_plain = df.apply(lambda x: pd.Series({"source": f"<cls>predict tail: {x['head']}, {x['head_description']}. {x['relation']}",
        #                                              "target": f"{x['tail']}",
        #                                              "relation": x['relation'],
        #                                              "tail": x['tail'],
        #                                              "tail_description": x['tail_description'],
        #                                              "head": x['head'],
        #                                              "head_description": x['head_description'],
        #                                              "objective": "predict_tail"
        #                                              }), axis=1).head(100)

        def randomly_delete_words(text):
            words = text.split()
            n_to_delete = len(words) // 10
            words_to_delete = np.random.choice(words, n_to_delete, replace=False)
            return ' '.join(word for word in words if word not in words_to_delete)
        
        # mem_tail_plain_deletes = df.apply(lambda x: pd.Series({"source": f"<cls>predict tail: {x['head']}, {x['head_description']}. {x['relation']}",
        #                                              "target": f"{x['tail']}",
        #                                              "relation": x['relation'],
        #                                              "tail": x['tail'],
        #                                              "tail_description": x['tail_description'],
        #                                              "head": x['head'],
        #                                              "head_description": x['head_description'],
        #                                              "objective": "predict_tail"
        #                                              }), axis=1)
        
        # mem_tail_plain_deletes['source'] = mem_tail_plain_deletes['source'].apply(randomly_delete_words)

        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # try:
        #     # Run the asynchronous function using the event loop
        #     mem_tail_with_questions = loop.run_until_complete(self.generate_questions(model, mem_tail))
        # finally:
        #     # Close the loop to clean up
        #     loop.close()

        mem_relation = df.apply(lambda x: pd.Series({"source": f"<cls>predict relation: Head: {x['head']}<head_description>{x['head_description']}</head_description>Relation: <relation>. Tail: {x['tail']}<tail_description>{x['tail_description']}</tail_description>",
                                                     "target": f"<relation>{x['relation']}<end>",
                                                     "relation": x['relation'],
                                                     "tail": x['tail'],
                                                     "tail_description": x['tail_description'],
                                                     "head": x['head'],
                                                     "head_description": x['head_description'],
                                                     "objective": "predict_relation"
                                                     }), axis=1)
        
        def process_row15(x):
            source, target = self.generate_word_masks(x['head'], x['head_description'], x['relation'], x['tail'], x['tail_description'], 15)
            return pd.Series({"source": source, 
                                "target": target,
                                "relation": x['relation'],
                                "tail": x['tail'],
                                "tail_description": x['tail_description'],
                                "head": x['head'],
                                "head_description": x['head_description'],
                                "objective": "mlm"})
        

        
        #mlm_df15 = df.apply(process_row15, axis=1)

        # Combine both transformations and remove duplicates
        new_df = pd.concat([mem_head, mem_relation, mem_tail]).drop_duplicates().reset_index(drop=True)
        return new_df

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
            | RunnableLambda(lambda x: "<cls>" + x + " Tail: <tail>") #RunnableLambda(lambda x: "<cls>Predict tail: Head and relation: " + x + " Tail: <tail>")
        )
        return chain

    async def async_invoke(self, question_generation_chain, head, relation):
        # Simulate an async API call
        # This function should be truly asynchronous and handle API calls or async operations
        return await question_generation_chain.ainvoke({"head": head, "relation": relation})
    
    async def generate_questions(self, model: ChatOpenAI, df):
        question_generation_chain = self.build_question_creation_chain(model)
        # Use asyncio to gather results from asynchronous operations
        # limit = 3
        # df_topk = df.head(limit).reset_index(drop=True)
        tasks = []
        for index, row in  tqdm(df.iterrows(), total=len(df), desc="Processing row in generate_questions"):
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

        self.vector_limit = 1000

        self.num_predictions = 60 # This is something we may want to vary by config
        self.max_length = 512
        self.max_output_length = 40

        self.metadata = []

        # Variables to track success score and total attempts for accuracy calculation
        self.address_success_score = 0
        self.address_total = 0
        
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

    # def prepare_query_vector(self, query_text):
    #     tokenized_query = self.tokenizer(query_text, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
    #     input_ids = tokenized_query['input_ids'].to('cuda')
    #     attention_mask = tokenized_query['attention_mask'].to('cuda')
    #     with torch.no_grad():  # Ensure model is in eval mode and no gradients are computed
    #         outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    #         #outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    #         last_hidden_state = outputs.last_hidden_state

    #         # Extract the first token from the decoder's outputs (assuming batch_first=True)
    #         embeddings = last_hidden_state[:, 0, :].cpu().numpy() 
    #     #return pooled_embeddings.cpu().numpy()  # # Pooling over the sequence dimension. Then, convert tensor to numpy array
    #     return embeddings

    def prepare_query_vector(self, query_text):
        tokenized_query = self.tokenizer(query_text, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
        input_ids = tokenized_query['input_ids'].to('cuda')
        attention_mask = tokenized_query['attention_mask'].to('cuda')
        with torch.no_grad():  # Ensure model is in eval mode and no gradients are computed
            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = encoder_outputs.last_hidden_state
            pooled_embeddings = torch.mean(embeddings, dim=1)  # Pooling over the sequence dimension
        return pooled_embeddings.cpu().numpy()  # Convert tensor to numpy array
    
    def write_to_faiss(self, batch_size=20):
        # Need to clear the index.
        self.faiss_index = faiss.IndexHNSWFlat(vector_length, self.index_m)
        self.faiss_index.hnsw.efConstruction = self.efConstruction
        self.metadata = []

        # Goal is to get validation questions to beat embeddings from training data
        # Combine and deduplicate entries from all datasets
        combined_df = pd.concat([self.train_dataset.df, self.val_dataset.df, self.test_dataset.df]).drop_duplicates().reset_index(drop=True)
        # Filter to only rows where 'objective' is 'predict_tail'
        filtered_df = combined_df[combined_df['objective'] == "predict_tail"].reset_index(drop=True)
        
        # Calculate the number of batches
        num_batches = len(filtered_df) // batch_size + (0 if len(filtered_df) % batch_size == 0 else 1)

        # Process each batch
        for batch_idx in range(num_batches):
            # Slice the DataFrame to get the batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_df = filtered_df.iloc[start_idx:end_idx]
            
            # Prepare batch data
            sources = batch_df['source'].tolist()
            tokenized_input = self.tokenizer(sources, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
            input_ids = tokenized_input['input_ids'].to('cuda')
            attention_mask = tokenized_input['attention_mask'].to('cuda')

            # Get encoder outputs
            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = encoder_outputs.last_hidden_state
            
            # Pool embeddings over the sequence dimension
            pooled_embeddings = torch.mean(embeddings, dim=1).cpu().numpy()  # Convert to numpy array for FAISS
            
            # Add embeddings to FAISS index in batch
            self.faiss_index.add(pooled_embeddings)

            # Store metadata associated with embeddings
            metadata_entries = batch_df.apply(lambda row: (row['objective'], row['head'], row['head_description'],
                                                        row['relation'], row['tail'], row['tail_description']), axis=1).tolist()
            self.metadata.extend(metadata_entries)

        print("FAISS indexing complete and metadata stored.")

    # def write_to_faiss(self, batch_size=20):
    #     # Need to clear the index.
    #     self.faiss_index = faiss.IndexHNSWFlat(vector_length, self.index_m)
    #     self.faiss_index.hnsw.efConstruction = self.efConstruction
    #     self.metadata = []

    #     # Goal is to get validation questions to beat embeddings from training data
    #     # Combine and deduplicate entries from all datasets
    #     combined_df = pd.concat([self.train_dataset.df, self.val_dataset.df]).drop_duplicates().reset_index(drop=True)
    #     # Filter to only rows where 'objective' is 'predict_tail'
    #     filtered_df = combined_df[combined_df['objective'] == "predict_tail"].reset_index(drop=True)
        
    #     # Calculate the number of batches
    #     num_batches = len(filtered_df) // batch_size + (0 if len(filtered_df) % batch_size == 0 else 1)

    #     # Process each batch
    #     for batch_idx in range(num_batches):
    #         # Slice the DataFrame to get the batch
    #         start_idx = batch_idx * batch_size
    #         end_idx = start_idx + batch_size
    #         batch_df = filtered_df.iloc[start_idx:end_idx]
            
    #         # Prepare batch data
    #         sources = batch_df['source'].tolist()
    #         tokenized_input = self.tokenizer(sources, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
    #         input_ids = tokenized_input['input_ids'].to('cuda')
    #         attention_mask = tokenized_input['attention_mask'].to('cuda')

    #         # Get outputs
    #         outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    #         #outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    #         last_hidden_state = outputs.last_hidden_state

    #         # Extract the first token from the decoder's outputs (assuming batch_first=True)
    #         embeddings = last_hidden_state[:, 0, :].cpu().numpy()  # Convert to numpy array for FAISS

    #         #embeddings = encoder_outputs.last_hidden_state
            
    #         # Pool embeddings over the sequence dimension
    #         # embeddings = torch.mean(embeddings, dim=1).cpu().numpy()  # Convert to numpy array for FAISS
            
    #         # Add embeddings to FAISS index in batch
    #         self.faiss_index.add(embeddings)

    #         # Store metadata associated with embeddings
    #         metadata_entries = batch_df.apply(lambda row: (row['objective'], row['head'], row['head_description'],
    #                                                     row['relation'], row['tail'], row['tail_description']), axis=1).tolist()
    #         self.metadata.extend(metadata_entries)

    #     print("FAISS indexing complete and metadata stored.")

    # common function for test and val evaluation
    def compute_vector_search_mrr(self, dataset, ranks_dicts):
        
        filtered_df = dataset.df[ (dataset.df['objective'] == 'predict_tail') & (dataset.df['test_question'].notna())].reset_index(drop=True)
        print(f"ANN df is length: {len(filtered_df)}")
        
        self.write_to_faiss()
        for index, row in filtered_df.iterrows():
            question = row["test_question"]
            true_tail = row["tail"]
        
            question_vector = self.prepare_query_vector(question)
        
            distances, indices = self.faiss_index.search(question_vector.reshape(1, -1), self.vector_limit)  # Reshape query to 2D array for FAISS
            results = [self.metadata[i][4] for i in indices[0]]  # Item with position==4 is the tail
        
            # Determine the rank of the true tail
            try:
                rank = next(i for i, result in enumerate(results, start=1) if result == true_tail)
                # Start = 1, so index position is already adjusted
            except StopIteration:
                # If the true tail is not in the results, you can decide how to handle this.
                # Options: skip, add a large number, or treat as the lowest rank (e.g., 10+1 if top 10 results).
                 # or 1 / (10 + 1) for last place beyond results
                length_of_index = self.faiss_index.ntotal
                rank = length_of_index + 1
            
            ranks_dict = defaultdict(list)
            ranks_dict["ann_ranks"].append(rank)
        ranks_dicts.append(ranks_dict)
        return ranks_dicts

    def clean_text(self, text):
        # if text == "<tail>splash, water and dust resistant</tail>": # Used just for testing
        #     return "splash, water, and dust resistant"
        # Regex pattern to find anything between < and >
        pattern = '<.*?>'
        # Replace the matched text with an empty string
        cleaned_text = re.sub(pattern, '', text)
        stripped = cleaned_text.strip()
        return stripped

    def compute_model_mrr(self, batch, dataset, ranks_dicts):
        input_batch = {
            'input_ids': batch['source_token_ids'], 
            'attention_mask': batch['attention_mask'],
            'temperature': 1,  # TODO: make this argument?
            'do_sample': True,
            'num_return_sequences': self.num_predictions,
            'num_beams': 1,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'max_length': self.max_output_length,
            'output_scores': True,
            'return_dict_in_generate': True,
            'max_length': 40
        }
       
        outputs = self.forward(source_token_ids=batch['source_token_ids'], 
                            attention_mask=batch['attention_mask'], 
                            target_token_ids=batch['target_token_ids'])
        loss = outputs.loss
        outputs = self.model.generate(**input_batch)#, max_new_tokens=128) # Run through decoder layer for evaluation
        # for pred in outputs[0]:
        #     print(self.tokenizer.decode(pred, skip_special_tokens=True))
        sequences = outputs.sequences
        # sequences has row count = self.num_predictions x batch_size
        # scores has tensor for each character. Each tensor is (self.num_predictions x batch_size) in rows, vocab_size in columns
        #print(f"Sequences: {sequences}")
        predictions: List[str] = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        ranks_dict = defaultdict(list)
        target_token_ids = batch["target_token_ids"]
        objectives = batch["objective"]
        heads = batch["head"]
        head_descriptions = batch["head_description"]
        relation = batch["relation"]
        tails = batch["tail"]
        tail_descriptions = batch["tail_description"]
        sources = batch["source"]
        targets = batch["target"]
        

        for idx, (source, target, objective) in enumerate(zip(sources, targets, objectives)):
            predictions_for_batch = predictions[idx * self.num_predictions : (idx + 1) * self.num_predictions]
            cleaned_predictions = [self.clean_text(pred) for pred in predictions_for_batch]
            preds = np.array(cleaned_predictions)
            true_pos = (preds == self.clean_text(target)).nonzero()[0]
            if len(true_pos) == 0:
                ranks_dict["model_ranks"].append(dataset.num_entities)
                if objective == "predict_tail":
                    ranks_dict["tail_ranks"].append(dataset.num_entities)
                elif objective == "predict_head":
                    ranks_dict["head_ranks"].append(dataset.num_entities)
                elif objective == "predict_relation":
                    ranks_dict["relation_ranks"].append(dataset.num_entities)
                elif objective == "mlm":
                    ranks_dict["mlm_ranks"].append(dataset.num_entities)
                else:
                    raise NotImplementedError("evaluating other objectives not implemented yet")
                ranks_dicts.append(ranks_dict)
            else:
                scores = outputs.scores
                scores = self.get_scores(sequences, scores)
                true_pos = true_pos[0] # We only need the first logprob since they're all identical for the same text
                true_score = scores[idx * self.num_predictions + true_pos]
                unique_preds, unique_indices = np.unique(preds, return_index=True)
                relevant_scores = scores[idx * self.num_predictions : (idx + 1) * self.num_predictions][unique_indices] # i.e. get the unique scores from slice for this batch
                # true_answers = self.dataset.filter_dict[source] # true_answers would be to check aliases as well
                rank = 0
                ties = 0
                for prediction, score in zip(unique_preds.tolist(), relevant_scores.tolist()):
                    if self.clean_text(prediction) == self.clean_text(target): # if p in true_answers: # true_answers would be to check aliases as well. Same with commented out line below.
                        continue
                    # if self.dataset.entity_inverse_alias_dict.get(prediction, None) is None:
                    #     continue
                    if score > true_score: # This means some value was ranked higher than our match (based on their logprobs)
                        rank += 1
                    if score == true_score:
                        ties += 1

                ranks_dict["model_ranks"].append(rank + ties // 2 + 1)
                if objective == "predict_tail":
                    ranks_dict["tail_ranks"].append(rank + ties // 2 + 1)
                elif objective == "predict_head":
                    ranks_dict["head_ranks"].append(rank + ties // 2 + 1)
                elif objective == "predict_relation":
                    ranks_dict["relation_ranks"].append(rank + ties // 2 + 1)
                elif objective == "mlm":
                    ranks_dict["mlm_ranks"].append(rank + ties // 2 + 1)
                else:
                    raise NotImplementedError("evaluating other objectives not implemented yet")
                ranks_dicts.append(ranks_dict)
        return ranks_dicts

    def evaluate(self, batch, mode='val'):
        ranks_dicts = []

        if mode == "val":
            dataset = self.val_dataset
        elif mode == "test":
            dataset = self.test_dataset
        elif mode == "train":
            dataset = self.train_dataset

        #ranks_dicts = self.compute_vector_search_mrr(dataset, ranks_dicts)
        ranks_dicts = self.compute_model_mrr(batch, dataset, ranks_dicts)

        self.ranks_dicts = ranks_dicts
        return ranks_dicts
            # compute mrr from how closely the results matched tail



        # Derived from: https://github.com/uma-pi1/kgt5-context/blob/main/kgt5_model.py
        

        # parsing the input
        

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
    
    # def on_validation_epoch_end(self):
    #     return self.metric_aggregation(self.ranks_dicts)

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
        return self.evaluate(batch, mode='test')

    def metric_aggregation(self, ranks_dicts):
        ranks = np.array([rd["model_ranks"] for rd in self.ranks_dicts]).squeeze()
        head_ranks = np.array([rd["head_ranks"] for rd in self.ranks_dicts if len(rd["head_ranks"]) > 0]).squeeze()
        tail_ranks = np.array([rd["tail_ranks"] for rd in self.ranks_dicts if len(rd["tail_ranks"]) > 0]).squeeze()
        #mlm_ranks = np.array([rd["mlm_ranks"] for rd in self.ranks_dicts if len(rd["mlm_ranks"]) > 0]).squeeze()
        relation_ranks = np.array([rd["relation_ranks"] for rd in self.ranks_dicts if len(rd["relation_ranks"]) > 0]).squeeze()
        ann_ranks = np.array([rd["ann_ranks"] for rd in self.ranks_dicts if len(rd["ann_ranks"]) > 0]).squeeze()

        for r, suffix in zip([ranks, head_ranks, tail_ranks, relation_ranks, ann_ranks], ["", "_head", "_tail", "_relation", "_ann"]):
            if len(r) != 0:
                mrr = np.mean(1/r).item()
                h1 = np.mean(r <= 1).item()
                h3 = np.mean(r <= 3).item()
                h10 = np.mean(r <= 10).item()
            else:
                mrr = 0.0
                h1 = 0.0
                h3 = 0.0
                h10 = 0.0
            self.log(f"mrr{suffix}", mrr)
            self.log(f"h1{suffix}", h1)
            self.log(f"h3{suffix}", h3)
            self.log(f"h10{suffix}", h10)
            print(f"\nmrr{suffix}", mrr)
            print(f"h1{suffix}", h1)
            print(f"h3{suffix}", h3)
            print(f"h10{suffix}", h10)

    def on_test_epoch_end(self):
        return self.metric_aggregation(self.ranks_dicts)


def main():
    # Used at eval
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    tokenizer.add_tokens(special_tokens)  
    data_module = DataModuleWith_Tail_NoMLM(tokenizer, batch_size=batch_size)
    data_module.setup()
    
    model = KBModel(data_module=data_module, special_tokens=special_tokens)
    #model = KBModel.load_from_checkpoint("/teamspace/jobs/t5-knowledge-graph-pretrain-v3-predict-tail-no-mlm-actually-pretrained/nodes.0/kg_logs_v4/knowledge_tuples_T5_knowledge_graph_pretrain_v3_predict_tail_no_mlm/version_0/checkpoints/epoch=2-step=972.ckpt", data_module=data_module, special_tokens=special_tokens)
    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger("kg_logs_v4", name=filename + "_" + suffix)

    
    # Use PyTorch Lightning's Trainer
    trainer = pl.Trainer(max_epochs=50, accelerator="gpu", devices=-1,
                        enable_progress_bar=True,
                        callbacks=[ModelCheckpoint(monitor="val_loss"),
                                    EarlyStopping(monitor="val_loss", patience=3)],
                        logger=logger,
                        precision="bf16-mixed")

    trainer.fit(model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()