import asyncio
import random
import re
import unittest
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from cassandra.concurrent import execute_concurrent_with_args
from transformers import T5Tokenizer

from core.ConfigLoader import ConfigLoader

from pydantic import BaseModel

seed = 42
random.seed(seed)
np.random.seed(seed)

context_text_length_max = 240
relation_name_length_max = 25
entity_name_length_max = 40
entity_description_length_max = 200
group_size_standard_deviation_max = 3

class TestStudy(unittest.TestCase):

    def read_wikidata5m_files(self, split_name: str):
        relations = pd.read_csv('data/relation_mentions.del', header=None, names=['RelationID', 'RelationName'],
                                sep='\t')
        entity_descriptions = pd.read_csv('data/entity_desc.del', header=None, names=['EntityID', 'EntityDescription'],
                                          sep='\t', index_col='EntityID')
        entity_names = pd.read_csv('data/entity_strings.del', header=None, names=['QID', 'EntityName'], sep='\t')
        entity_ids = pd.read_csv('data/entity_ids.del', header=None, names=['EntityID', 'QID'], sep='\t')
        triple_ids = pd.read_csv(f"data/{split_name}.del", header=None, names=['HeadEntityID', 'RelationID', 'TailEntityID'],
                                 sep='\t')
        return relations, entity_ids, entity_names, entity_descriptions, triple_ids

    def get_fake_wikidata(self):
        from io import StringIO

        # Simulating CSV data
        relations_data = StringIO("""
RelationID\tRelationName
1\tpartners with
2\tacquired by
3\tcompetes with
4\tmerged with
5\tsold products to
                """)
        entity_descriptions_data = StringIO("""
EntityID\tEntityDescription
1\tTech firm
2\tStartup that works with Company C. Similar to company F
3\tRetailer
4\tManufacturer
5\tE-commerce company, a company that worked with Company A
6\tTech firm, the parent of Company A
                """)
        entity_names_data = StringIO("""
QID\tEntityName
101\tCompany A
102\tCompany B
103\tCompany C
104\tCompany D
105\tCompany E
106\tCompany F
                """)
        entity_ids_data = StringIO("""
EntityID\tQID
1\t101
2\t102
3\t103
4\t104
5\t105
6\t106
                """)
        triple_ids_data = StringIO("""
HeadEntityID\tRelationID\tTailEntityID
1\t1\t4
2\t2\t5
3\t3\t6
1\t4\t4
4\t2\t6
1\t2\t6
6\t3\t2
2\t1\t3
3\t1\t5
3\t1\t2
6\t5\t1
6\t5\t3
3\t3\t4
                """)

        # Reading simulated data
        relations = pd.read_csv(relations_data, sep='\t')
        entity_descriptions = pd.read_csv(entity_descriptions_data, sep='\t', index_col='EntityID')
        entity_names = pd.read_csv(entity_names_data, sep='\t')
        entity_ids = pd.read_csv(entity_ids_data, sep='\t')
        triple_ids = pd.read_csv(triple_ids_data, sep='\t')
        return relations, entity_ids, entity_names, entity_descriptions, triple_ids

    def filter_by_deviation(self, df, column_name, deviations: int):
        """
        deviations is to filter by that many standard deviations from the mean
        """
        pd.set_option('display.float_format', '{:.2f}'.format)
        # Ensure the column exists in the dataframe
        if column_name not in df.columns:
            print(f"Column {column_name} not found in the dataframe.")
            return None

        # Compute basic statistics of the column
        data = df[column_name].str.len()
        print("Initial Statistics:")
        print(data.describe())

        # Calculate standard deviation from the mean
        mean = data.mean()
        std_dev = data.std()
        upper_limit = mean + deviations * std_dev

        # Filtering out values greater than N standard deviations above the mean
        filtered_df = df[data <= upper_limit].reset_index(drop=True)

        return filtered_df
    def analyze_and_filter(self, df, column_name, deviations: int):
        """
        deviations is to filter by that many standard deviations from the mean
        """
        pd.set_option('display.float_format', '{:.2f}'.format)
        # Ensure the column exists in the dataframe
        if column_name not in df.columns:
            print(f"Column {column_name} not found in the dataframe.")
            return None

        # Compute basic statistics of the column
        data = df[column_name].str.len()
        print("Initial Statistics:")
        print(data.describe())

        # Plotting histogram of the initial data
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(data, bins=30, color='blue', edgecolor='black')
        plt.title(f'Histogram of Original Data for column {column_name}')
        plt.xlabel(column_name + " character length")
        plt.ylabel('Frequency')

        # Plotting histogram on a log scale
        plt.subplot(1, 2, 2)
        plt.hist(data, bins=30, color='green', edgecolor='black', log=True)
        plt.title(f'Log Scale Histogram of Original Data for column {column_name}')
        plt.xlabel(column_name + " character length")
        plt.ylabel('Log Frequency')
        plt.show()

        # Calculate 4th standard deviation from the mean
        mean = data.mean()
        std_dev = data.std()
        upper_limit = mean + deviations * std_dev

        print(f"\n4th standard deviation (upper limit): {upper_limit}")

        # Filtering out values greater than 4 standard deviations above the mean
        filtered_df = df[data <= upper_limit].reset_index(drop=True)
        filtered_data = filtered_df[column_name].str.len()

        # Display statistics of the filtered data
        print(f"\nFiltered Data Statistics for column {column_name}:")
        print(filtered_data.describe())

        # Plotting histograms of the filtered data
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(filtered_data, bins=30, color='blue', edgecolor='black')
        plt.title(f'Histogram of Filtered Data for column {column_name}')
        plt.xlabel(column_name + " character length")
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        plt.hist(filtered_data, bins=30, color='green', edgecolor='black', log=True)
        plt.title(f'Log Scale Histogram of Filtered Data for column {column_name}')
        plt.xlabel(column_name + " character length")
        plt.ylabel('Log Frequency')
        plt.show()

        return filtered_df

    def filter_by_group(self, df, groupby_column, count_column, deviations: int):

        pd.set_option('display.float_format', '{:.2f}'.format)
        # Group by the specified column and count the number of occurrences
        group_counts = df.groupby(groupby_column)[count_column].count()

        # Set a threshold for filtering, here using 4 standard deviations as an example
        mean_count = group_counts.mean()
        std_dev_count = group_counts.std()
        upper_limit = mean_count + deviations * std_dev_count

        # Find groups that meet the criterion
        valid_groups = group_counts[group_counts <= upper_limit].index

        # Filter the original DataFrame based on these groups
        filtered_df = df[df[groupby_column].isin(valid_groups)].reset_index(drop=True)
        return filtered_df
    def analyze_and_filter_by_group(self, df, groupby_column, count_column, deviations: int):
        pd.set_option('display.float_format', '{:.2f}'.format)
        # Group by the specified column and count the number of occurrences
        group_counts = df.groupby(groupby_column)[count_column].count()

        # Set a threshold for filtering, here using 4 standard deviations as an example
        mean_count = group_counts.mean()
        std_dev_count = group_counts.std()
        upper_limit = mean_count + deviations * std_dev_count

        # Find groups that meet the criterion
        valid_groups = group_counts[group_counts <= upper_limit].index

        # Filter the original DataFrame based on these groups
        filtered_df = df[df[groupby_column].isin(valid_groups)].reset_index(drop=True)

        print("\nFiltered group counts:")
        filtered_group_counts = group_counts[group_counts <= upper_limit]
        print(filtered_group_counts.describe())

        # Print initial and filtered statistics
        print("Initial group counts:")
        print(group_counts.describe())
        # Plotting histogram of the initial data
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(group_counts.values, bins=30, color='blue', edgecolor='black')
        plt.title(f'Histogram of Initial Group Counts')
        plt.xlabel("Group counts")
        plt.ylabel('Frequency')

        # Plotting histogram on a log scale
        plt.subplot(1, 2, 2)
        plt.hist(group_counts.values, bins=30, color='green', edgecolor='black', log=True)
        plt.title(f'Log Scale Histogram of Initial Group Counts')
        plt.xlabel("Group counts")
        plt.ylabel('Log Frequency')
        plt.show()

        # Plotting histogram of the initial data
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(filtered_group_counts.values, bins=30, color='blue', edgecolor='black')
        plt.title(f'Histogram of Filtered Group Counts')
        plt.xlabel(groupby_column + " counts")
        plt.ylabel('Frequency')

        # Plotting histogram on a log scale
        plt.subplot(1, 2, 2)
        plt.hist(filtered_group_counts.values, bins=30, color='green', edgecolor='black', log=True)
        plt.title(f'Log Scale Histogram of Filtered Group Counts')
        plt.xlabel(groupby_column + " counts")
        plt.ylabel('Log Frequency')
        plt.show()

        return filtered_df

    def compute_stats(self):
        split = "train"
        relations, entity_ids, entity_names, entity_descriptions, triple_ids = self.read_wikidata5m_files(split)

        relations = self.analyze_and_filter(relations, "RelationName", 3)
        entity_descriptions = self.analyze_and_filter(entity_descriptions, 'EntityDescription', 3)
        entity_names = self.analyze_and_filter(entity_names, "EntityName", 3)
        triple_ids = self.analyze_and_filter_by_group(triple_ids, "HeadEntityID", 'RelationID', 3)

    def filter_triples_logically_before_join(self, triple_df):
        # Create masks for each condition
        condition1 = triple_df.apply(
            lambda row: row['EntityName_Tail'].lower() in row['EntityName_Head'].lower(), axis=1
        )
        condition2 = triple_df.apply(
            lambda row: row['EntityName_Head'].lower() in row['EntityName_Tail'].lower(), axis=1
        )
        # Use apply to check substring presence, row by row
        condition3 = triple_df.apply(
            lambda row: row['EntityName_Tail'].lower() in row['EntityDescription_Head'].lower(), axis=1)
        condition4 = triple_df.apply(
            lambda row: row['EntityName_Head'].lower() in row['EntityDescription_Tail'].lower(), axis=1)


        # Combine conditions using OR (|)
        filter_condition = condition1 | condition2 | condition3 | condition4
        # Apply the filter with a negation to keep entries that do not meet any of the conditions
        filtered_triple_ids = triple_df[~filter_condition]

        return filtered_triple_ids.reset_index(drop=True)

    def build_full_dataset(self, relation_df, entity_description_df, entity_name_df, entity_id_df, triple_id_df):
        relation_df, entity_description_df, entity_name_df, triple_id_df = self.trim_statistically(relation_df, entity_description_df, entity_name_df, triple_id_df)
        triple_df = self.build_triples_with_descriptions(entity_description_df, entity_id_df, entity_name_df, relation_df,
                                                         triple_id_df)
        filtered_df = self.filter_triples_logically_before_join(triple_df)

        # def process_group(group):
        #     # I need to do the cross-join within the group's apply to limit the size of the cartesian product

        # Create a temporary key column for cross joining within each group
        filtered_df['key'] = 1
        merged_parent_child_df = pd.merge(filtered_df, filtered_df, on=['HeadEntityID', 'key'], suffixes=('_parent', '_child'))

        # Drop the temporary 'key' column as it's no longer needed after merging
        merged_parent_child_df.drop('key', axis=1, inplace=True)
        filtered_df.drop('key', axis=1, inplace=True)
        print(merged_parent_child_df.iloc[8])

        filtered_parent_child_df = self.filter_parent_child_pairs_logically(merged_parent_child_df)
        grouped_df = filtered_parent_child_df.groupby("TailEntityID_parent")
        def build_context(group):
            concatenated_result = ', '.join(group['RelationName_child'] + ' (' + group['EntityName_Tail_child'] + ')')
            # Concatenate 'RelationName' and 'EntityName_Tail' with conditions
            result_with_wrapper = "<context>" + concatenated_result[:context_text_length_max] + "</context>" if concatenated_result else ''
            return pd.DataFrame({'HeadEntityID': group["HeadEntityID"].iloc[0], 'Context': [result_with_wrapper]})  # group.name assumes grouping by HeadEntityID
        context_with_ids_df = grouped_df.apply(build_context).reset_index(drop=True)

        # Rename the 'Context' column to 'HeadContext' for use in the first merge
        context_with_head_ids_df = context_with_ids_df.rename(columns={'Context': 'HeadContext'})

        # Perform the first merge
        filtered_df_with_head_context = filtered_df.merge(context_with_head_ids_df, on="HeadEntityID", how="left")

        # Rename the 'Context' column to 'TailContext' for use in the second merge
        context_with_tail_ids_df = context_with_ids_df.rename(columns={'Context': 'TailContext', "HeadEntityID": "HeadEntityID_context"})

        # Perform the second merge
        filtered_df_with_head_and_tail_context = filtered_df_with_head_context.merge(context_with_tail_ids_df, left_on="TailEntityID", right_on="HeadEntityID_context", how="left")
        filtered_df_with_head_and_tail_context.drop('HeadEntityID_context', axis=1, inplace=True)
        filtered_df_with_head_and_tail_context.fillna('', inplace=True)

        import numpy as np

        # Vectorized function to check substring presence
        def check_substring(company_name, context):
            # Convert all elements to string ensuring no non-string types cause issues
            str_context = np.array(list(map(str, context)))
            str_company_name = np.array(list(map(str, company_name)))
            result = np.char.find(str_context,str_company_name)
            return result == -1  # Returns True where substring is not found

        # Create mask for 'EntityName_Tail' in 'HeadContext'
        condition1 = check_substring(filtered_df_with_head_and_tail_context['EntityName_Tail'], filtered_df_with_head_and_tail_context['HeadContext'])

        # Create mask for 'EntityName_Head' in 'TailContext'
        condition2 = check_substring(filtered_df_with_head_and_tail_context['EntityName_Head'], filtered_df_with_head_and_tail_context['TailContext'])

        combined_conditions = condition1 & condition2
        # Apply the masks using negation to filter out those rows
        filtered_df_with_head_and_tail_context_filtered = filtered_df_with_head_and_tail_context[combined_conditions]
        return filtered_df_with_head_and_tail_context_filtered

    def filter_parent_child_pairs_logically(self, merged_parent_child_df):
        # Now, we apply the filter criteria. We will roll up the child tails when neither the parent head nor parent tail
        # appears in it so we don't leak information from source to target

        #condition1 = merged_parent_child_df["TailEntityID_parent"] == merged_parent_child_df["TailEntityID_child"]

        # exclude cases where:
        #   child tail's name exists in parent head's or parent tail's names or descriptions
        #   parent head's name exists in child tail's name or descriptions
        #   parent tail's name exists in child tail's name or descriptions
        condition2 = merged_parent_child_df.apply(
            lambda row: row['EntityName_Tail_child'].lower() in row['EntityName_Head_parent'].lower(), axis=1
        )
        condition3 = merged_parent_child_df.apply(
            lambda row: row['EntityName_Tail_child'].lower() in row['EntityDescription_Head_parent'].lower(), axis=1
        )
        # condition4 = merged_parent_child_df.apply(
        #     lambda row: row['EntityName_Tail_child'].lower() in row['EntityName_Tail_parent'].lower(), axis=1
        # )
        condition5 = merged_parent_child_df.apply(
            lambda row: row['EntityName_Tail_child'].lower() in row['EntityDescription_Tail_parent'].lower(), axis=1
        )
        condition6 = merged_parent_child_df.apply(
            lambda row: row['EntityName_Head_parent'].lower() in row['EntityName_Tail_child'].lower(), axis=1
        )
        condition7 = merged_parent_child_df.apply(
            lambda row: row['EntityName_Head_parent'].lower() in row['EntityDescription_Tail_child'].lower(), axis=1
        )
        # condition8 = merged_parent_child_df.apply(
        #     lambda row: row['EntityName_Tail_parent'].lower() in row['EntityName_Tail_child'].lower(), axis=1
        # )
        condition9 = merged_parent_child_df.apply(
            lambda row: row['EntityName_Tail_parent'].lower() in row['EntityDescription_Tail_child'].lower(), axis=1
        )
        filter_condition =  condition2 | condition3 | condition5 | condition6 | condition7 | condition9
        # Apply the filter with a negation to keep entries that do not meet any of the conditions
        filtered_merged_parent_child_df = merged_parent_child_df[~filter_condition]

        return filtered_merged_parent_child_df.reset_index(drop=True)

    def build_triples_with_descriptions(self, entity_description_df, entity_id_df, entity_name_df, relation_df, triple_id_df):
        # Merge entity information
        entities = pd.merge(entity_id_df, entity_name_df, on='QID')
        entities = entities.set_index('EntityID').join(entity_description_df)
        # Enhance triple_ids with entity names and descriptions
        triple_ids = triple_id_df.merge(entities.add_suffix('_Head'), left_on='HeadEntityID', right_index=True)
        triple_ids = triple_ids.merge(entities.add_suffix('_Tail'), left_on='TailEntityID', right_index=True)
        triple_ids = triple_ids.merge(relation_df, on='RelationID')
        triple_ids = triple_ids[
            ['HeadEntityID', 'QID_Head', 'EntityName_Head', 'EntityDescription_Head', 'RelationID', 'RelationName',
             'TailEntityID', 'QID_Tail', 'EntityName_Tail', 'EntityDescription_Tail']]
        # Fill NaN with empty strings and convert data to strings to avoid type issues
        columns_to_fill = ['RelationName', 'EntityName_Tail', 'EntityName_Head', 'EntityDescription_Tail',
                           'EntityDescription_Head']
        for col in columns_to_fill:
            triple_ids[col] = triple_ids[col].fillna('').astype(str)
        return triple_ids
    def trim_statistically(self, relations, entity_descriptions, entity_names, triple_ids):
        # relations = self.filter_by_deviation(relations, "RelationName", 3)
        # entity_descriptions = self.filter_by_deviation(entity_descriptions, 'EntityDescription', 3)
        # entity_names = self.filter_by_deviation(entity_names, "EntityName", 3)
        # triple_ids = self.filter_by_group(triple_ids, "HeadEntityID", 'RelationID', 3)
        # (decided to just truncate text instead of removing the rows entirely, except for the rare excessively large groups)
        relations["RelationName"] = relations["RelationName"].str[:relation_name_length_max]
        entity_names["EntityName"] = entity_names["EntityName"].str[:entity_name_length_max]
        entity_descriptions["EntityDescription"] = entity_descriptions["EntityDescription"].str[:entity_description_length_max]
        triple_ids = self.filter_by_group(triple_ids, "HeadEntityID", 'RelationID', group_size_standard_deviation_max)
        return relations, entity_descriptions, entity_names, triple_ids

    def build_training_labels(self, df):
        mem_head = df.apply(lambda x: pd.Series({"source": f"<cls>predict head: Head: <head>. Relation: {x['relation']}. Tail: {x['tail']}<tail_description>{x['tail_description']}</tail_description>",
                                                         "target": f"<head>{x['head']}<end>",
                                                         "relation": x['relation'],
                                                         "tail": x['tail'],
                                                         "tail_description": x['tail_description'],
                                                         "head": x['head'],
                                                         "head_description": x['head_description'],
                                                         "head_context": x['head_context'],
                                                         "tail_context": x['tail_context'],
                                                         "objective": "predict_head"
                                                         }), axis=1)
        mem_tail = df.apply(lambda x: pd.Series({"source": f"<cls>predict tail: Head: {x['head']}<head_description>{x['head_description']}</head_description>Relation: {x['relation']}. Tail: <tail>",
                                                         "target": f"<tail>{x['tail']}<end>",
                                                         "relation": x['relation'],
                                                         "tail": x['tail'],
                                                         "tail_description": x['tail_description'],
                                                         "head": x['head'],
                                                         "head_description": x['head_description'],
                                                         "head_context": x['head_context'],
                                                         "tail_context": x['tail_context'],
                                                         "objective": "predict_tail"
                                                         }), axis=1)

        mem_relation = df.apply(lambda x: pd.Series({"source": f"<cls>predict relation: Head: {x['head']}<head_description>{x['head_description']}</head_description>Relation: <relation>. Tail: {x['tail']}<tail_description>{x['tail_description']}</tail_description>",
                                                             "target": f"<relation>{x['relation']}<end>",
                                                             "relation": x['relation'],
                                                             "tail": x['tail'],
                                                             "tail_description": x['tail_description'],
                                                             "head": x['head'],
                                                             "head_description": x['head_description'],
                                                             "head_context": x['head_context'],
                                                             "tail_context": x['tail_context'],
                                                             "objective": "predict_relation"
                                                             }), axis=1)
        new_df = pd.concat([mem_head, mem_relation, mem_tail]).drop_duplicates().reset_index(drop=True)
        return new_df
    def test_entire_wikidata_pipeline_fake(self):
        split = "fake"
        relation_df, entity_id_df, entity_name_df, entity_description_df, triple_id_df = self.get_fake_wikidata()
        prepared_df = self.build_full_dataset(relation_df, entity_description_df, entity_name_df, entity_id_df, triple_id_df)
        prepared_df = prepared_df.rename(columns={
            'EntityName_Head': 'head',
            'EntityDescription_Head': 'head_description',
            'HeadContext': 'head_context',
            'RelationName': 'relation',
            'EntityName_Tail': 'tail',
            'EntityDescription_Tail': 'tail_description',
            'TailContext': 'tail_context'
        })
        training_set = self.build_training_labels(prepared_df)
        max_length = training_set['source'].apply(len).max()
        print(f"max_length is: {max_length}")
        shuffled_df = prepared_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        # tokenizer = self.get_tokenizer()
        # shuffled_df["source_tokenized"] = shuffled_df['source'].apply(lambda x: tokenizer.encode_plus(x, max_length=512, padding='max_length', truncation=True, return_tensors='pt')['input_ids'].squeeze().tolist())
        # shuffled_df["attention_mask"] = shuffled_df["source_tokenized"].apply(lambda x: (torch.tensor(x).ne(tokenizer.pad_token_id)).tolist())
        # shuffled_df["target_tokenized"] = shuffled_df['target'].apply(lambda x: tokenizer.encode_plus(x, max_length=512, padding='max_length', truncation=True, return_tensors='pt')['input_ids'].squeeze().tolist())

        print(shuffled_df.head())
        shuffled_df.to_parquet(f"data/wikidata5m_train_processed_filtered_{split}.parquet", engine='fastparquet')

    def test_build_all_wikidata_splits(self):
        print("Starting validation tiny data prep.")
        self.build_entire_wikidata_pipeline_real("valid_tiny")
        print("Starting validation data prep.")
        self.build_entire_wikidata_pipeline_real("valid")
        print("Starting test data prep.")
        self.build_entire_wikidata_pipeline_real("test")
        print("Starting training data prep.")
        #self.build_entire_wikidata_pipeline_real("train")

    def build_entire_wikidata_pipeline_real(self, split:str):
        relation_df, entity_id_df, entity_name_df, entity_description_df, triple_id_df = self.read_wikidata5m_files(split)
        prepared_df = self.build_full_dataset(relation_df, entity_description_df, entity_name_df, entity_id_df, triple_id_df)
        prepared_df = prepared_df.rename(columns={
            'EntityName_Head': 'head',
            'EntityDescription_Head': 'head_description',
            'HeadContext': 'head_context',
            'RelationName': 'relation',
            'EntityName_Tail': 'tail',
            'EntityDescription_Tail': 'tail_description',
            'TailContext': 'tail_context'
        })
        training_set = self.build_training_labels(prepared_df)
        max_length = training_set['source'].apply(len).max()
        print(f"max_length is: {max_length}")
        shuffled_df = prepared_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        # tokenizer = self.get_tokenizer()
        # shuffled_df["source_tokenized"] = shuffled_df['source'].apply(lambda x: tokenizer.encode_plus(x, max_length=512, padding='max_length', truncation=True, return_tensors='pt')['input_ids'].squeeze().tolist())
        # shuffled_df["attention_mask"] = shuffled_df["source_tokenized"].apply(lambda x: (torch.tensor(x).ne(tokenizer.pad_token_id)).tolist())
        # shuffled_df["target_tokenized"] = shuffled_df['target'].apply(lambda x: tokenizer.encode_plus(x, max_length=512, padding='max_length', truncation=True, return_tensors='pt')['input_ids'].squeeze().tolist())

        print(shuffled_df.head())
        shuffled_df.to_parquet(f"data/wikidata5m_train_processed_filtered_{split}.parquet", engine='fastparquet')

    def get_tokenizer(self):
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        special_tokens = [
            "<cls>", "<head>", "<head_description>", "</head_description>", "<mask>","<relation>", "<tail>", "<tail_description>","</tail_description>","<relation>", "<end>"
        ]
        tokenizer.add_tokens(special_tokens)
        return tokenizer

    def test_filter_child_pairs_logically_handles_singles(self):

        data = {
            'HeadEntityID': [4],
            'QID_Head_parent': [104],
            'EntityName_Head_parent': ['Company D'],
            'EntityDescription_Head_parent': ['Manufacturer'],
            'RelationID_parent': [2],
            'RelationName_parent': ['acquired by'],
            'TailEntityID_parent': [6],
            'QID_Tail_parent': [106],
            'EntityName_Tail_parent': ['Company F'],
            'EntityDescription_Tail_parent': ['Tech firm, the parent of Company A'],
            'QID_Head_child': [104],
            'EntityName_Head_child': ['Company D'],
            'EntityDescription_Head_child': ['Manufacturer'],
            'RelationID_child': [2],
            'RelationName_child': ['acquired by'],
            'TailEntityID_child': [6],
            'QID_Tail_child': [106],
            'EntityName_Tail_child': ['Company F'],
            'EntityDescription_Tail_child': ['Tech firm, the parent of Company A']
        }

        # Creating the DataFrame
        df = pd.DataFrame(data)
        filtered_df = self.filter_parent_child_pairs_logically(df)
        assert len(filtered_df) > 0, "DataFrame is empty"

    def test_fake_wikidata_pipeline_data_are_correct(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        relation_df, entity_id_df, entity_name_df, entity_description_df, triple_id_df = self.get_fake_wikidata()
        prepared_df = self.build_full_dataset(relation_df, entity_description_df, entity_name_df, entity_id_df, triple_id_df)
        prepared_df = prepared_df.rename(columns={
            'EntityName_Head': 'head',
            'EntityDescription_Head': 'head_description',
            'HeadContext': 'head_context',
            'RelationName': 'relation',
            'EntityName_Tail': 'tail',
            'EntityDescription_Tail': 'tail_description',
            'TailContext': 'tail_context'
        })

        assert prepared_df["head"].iloc[0] == "Company A"
        assert prepared_df["relation"].iloc[0] == "partners with"
        assert prepared_df["tail"].iloc[0] == "Company D"
        assert prepared_df["head_context"].iloc[0] == ""
        assert prepared_df["tail_context"].iloc[0] == "" # because F has A in description

        assert prepared_df["head"].iloc[1] == "Company B"
        assert prepared_df["relation"].iloc[1] == "acquired by"
        assert prepared_df["tail"].iloc[1] == "Company E"
        assert prepared_df["head_context"].iloc[1] == "" # B has C and F in description, so it has no available context
        assert prepared_df["tail_context"].iloc[1] == ""

        assert prepared_df["head"].iloc[2] == "Company C"
        assert prepared_df["relation"].iloc[2] == "competes with"
        assert prepared_df["tail"].iloc[2] == "Company F"
        assert prepared_df["head_context"].iloc[2] == "<context>partners with (Company E), competes with (Company D)</context>"
        assert prepared_df["tail_context"].iloc[2] == ""

        assert prepared_df["head"].iloc[3] == "Company A"
        assert prepared_df["relation"].iloc[3] == "merged with"
        assert prepared_df["tail"].iloc[3] == "Company D"
        assert prepared_df["head_context"].iloc[3] == "" # F has A in description, so it's omitted
        assert prepared_df["tail_context"].iloc[3] == "" # F has A in description, so it's omitted

        assert prepared_df["head"].iloc[4] == "Company D"
        assert prepared_df["relation"].iloc[4] == "acquired by"
        assert prepared_df["tail"].iloc[4] == "Company F"
        assert prepared_df["head_context"].iloc[4] == ""
        assert prepared_df["tail_context"].iloc[4] == "<context>sold products to (Company C)</context>" # B omited since it has F in description. A omitted since it's in description of F.


        assert prepared_df["head"].iloc[5] == "Company C"
        assert prepared_df["relation"].iloc[5] == "partners with"
        assert prepared_df["tail"].iloc[5] == "Company E"
        assert prepared_df["head_context"].iloc[5] == "<context>competes with (Company F), competes with (Company D)</context>" # E omitted since E is in tail. B omitted since C is in its description
        assert prepared_df["tail_context"].iloc[5] == ""

        assert prepared_df["head"].iloc[6] == "Company F"
        assert prepared_df["head_description"].iloc[6] == "Tech firm, the parent of Company A"
        assert prepared_df["relation"].iloc[6] == "sold products to"
        assert prepared_df["tail"].iloc[6] == "Company C"
        assert prepared_df["head_context"].iloc[6] == "" # B omitted since C is in B's description. A omitted since it's in F's description.
        assert prepared_df["tail_context"].iloc[6] == "<context>partners with (Company E), competes with (Company D)</context>"

        assert prepared_df["head"].iloc[7] == "Company C"
        assert prepared_df["relation"].iloc[7] == "sold products to"
        assert prepared_df["tail"].iloc[7] == "Company D"
        assert prepared_df["head_context"].iloc[7] == "<context>competes with (Company F), partners with (Company E)</context>" # B omitted since C is in its description
        assert prepared_df["tail_context"].iloc[7] == "<context>acquired by (Company F)</context>"


    @DeprecationWarning
    def test_fake_wikidata_pipeline_can_read_serialized(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        relations, entity_ids, entity_names, entity_descriptions, triple_ids = self.get_fake_wikidata()
        new_df = self.prepare_fake_relation_data(relations, entity_ids, entity_names, entity_descriptions, triple_ids)
        shuffled_df = new_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        max_length = new_df['source'].apply(len).max()
        print(max_length)
        tokenizer = self.get_tokenizer()
        shuffled_df["source_tokenized"] = shuffled_df['source'].apply(lambda x: tokenizer.encode_plus(x, max_length=512, padding='max_length', truncation=True, return_tensors='pt')['input_ids'].squeeze().tolist())
        shuffled_df["attention_mask"] = shuffled_df["source_tokenized"].apply(lambda x: (torch.tensor(x).ne(tokenizer.pad_token_id)).tolist())
        shuffled_df["target_tokenized"] = shuffled_df['target'].apply(lambda x: tokenizer.encode_plus(x, max_length=512, padding='max_length', truncation=True, return_tensors='pt')['input_ids'].squeeze().tolist())

        print(shuffled_df.head())
        shuffled_df.to_parquet(f"data/wikidata5m_train_processed_fake.parquet", engine='fastparquet')

        # Now, test that we can read it back.
        written_df = pd.read_parquet(f"data/wikidata5m_train_processed_fake.parquet", engine="fastparquet")
        original_dtype = torch.long
        original_shape = [512]
        serialized_tensor = torch.tensor(written_df.iloc[0]['source_tokenized'])
        #print(tensor_restored)