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

class Part(BaseModel):
    text: str
    position: int
    token: str
    bucket: Optional[str] = None
    swapped: bool

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
        relation_df, entity_description_df, entity_name_df, triple_id_df = self.trim_statistically(self, relation_df, entity_description_df, entity_name_df, triple_id_df)
        triple_df = self.build_triples_with_descriptions(entity_description_df, entity_id_df, entity_name_df, relation_df,
                                                         triple_id_df)
        filtered_df = self.filter_triples_logically_before_join(triple_df)
        merged_parent_child_df = pd.merge(filtered_df, filtered_df, on='HeadEntityID', suffixes=('_parent', '_child'))
        # G
        filtered_parent_child_df = self.filter_parent_child_pairs_logically(merged_parent_child_df)
        grouped_df = filtered_parent_child_df.groupby("TailEntityID_parent")
        contextualized_df = grouped_df.apply(build_context)
        join back to filtered_df on both head and tail (separate left joins). Replace nan with ''
        return contextualized_df

    def filter_parent_child_pairs_logically(self, merged_parent_child_df):
        # Now, we apply the filter criteria. We will roll up the child tails when neither the parent head nor parent tail
        # appears in it so we don't leak information from source to target
        condition1 = merged_parent_child_df["TailEntityID_parent"] == merged_parent_child_df["TailEntityID_child"]

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
        condition4 = merged_parent_child_df.apply(
            lambda row: row['EntityName_Tail_child'].lower() in row['EntityName_Tail_parent'].lower(), axis=1
        )
        condition5 = merged_parent_child_df.apply(
            lambda row: row['EntityName_Tail_child'].lower() in row['EntityDescription_Tail_parent'].lower(), axis=1
        )
        condition6 = merged_parent_child_df.apply(
            lambda row: row['EntityName_Head_parent'].lower() in row['EntityName_Tail_child'].lower(), axis=1
        )
        condition7 = merged_parent_child_df.apply(
            lambda row: row['EntityName_Head_parent'].lower() in row['EntityDescription_Tail_child'].lower(), axis=1
        )
        condition8 = merged_parent_child_df.apply(
            lambda row: row['EntityName_Tail_parent'].lower() in row['EntityName_Tail_child'].lower(), axis=1
        )
        condition9 = merged_parent_child_df.apply(
            lambda row: row['EntityName_Tail_parent'].lower() in row['EntityDescription_Tail_child'].lower(), axis=1
        )
        filter_condition = condition1 | condition2 | condition3 | condition4 | condition5 | condition6 | condition7 | condition8 | condition9
        # Apply the filter with a negation to keep entries that do not meet any of the conditions
        filtered_merged_parent_child_df = merged_parent_child_df[~filter_condition]

        return filtered_merged_parent_child_df.reset_index(drop=True)
    @DeprecationWarning
    def prepare_wikidata5m_from_dfs_with_description_filter(self, relation_df, entity_description_df, entity_name_df, entity_id_df, triple_id_df):
        triple_ids = self.build_triples_with_descriptions(entity_description_df, entity_id_df, entity_name_df, relation_df,
                                                          triple_id_df)

        # Create masks for each condition
        condition1 = triple_ids['EntityName_Tail'] == triple_ids['EntityName_Head']
        # Use apply to check substring presence, row by row
        condition2 = triple_ids.apply(
            lambda row: row['EntityName_Tail'].lower() in row['EntityDescription_Head'].lower(), axis=1)
        condition3 = triple_ids.apply(
            lambda row: row['EntityName_Head'].lower() in row['EntityDescription_Tail'].lower(), axis=1)
        # Combine conditions using OR (|)
        filter_condition = condition1 | condition2 | condition3
        # Apply the filter with a negation to keep entries that do not meet any of the conditions
        filtered_triple_ids = triple_ids[~filter_condition]

        # Ensure that no concatenation results in 'nan'
        def custom_concat_head(group):
            # Fill NaN values in specific columns to avoid 'nan' in the output
            group['RelationName'] = group['RelationName'].fillna('')
            group['EntityName_Tail'] = group['EntityName_Tail'].fillna('')

            # Apply the filters based on conditions specified in the comments
            filtered_group = group[
                ~group['EntityName_Head'].isin(group['EntityName_Tail']) &
                ~group['EntityName_Head'].isin(group['EntityDescription_Tail']) &
                ~group['EntityName_Tail'].apply(lambda x: x in group['EntityDescription_Head'].iloc[0] or x == group['EntityName_Head'].iloc[0] or x == group['EntityName_Tail'].iloc[0])
                ]
            concatenated_result = ', '.join(filtered_group['RelationName'] + ' (' + filtered_group['EntityName_Tail'] + ')')
            # Concatenate 'RelationName' and 'EntityName_Tail' with conditions
            return "<context>" + concatenated_result[:240] + "</context>" if concatenated_result else ''

        # Group by HeadEntityID to concatenate relation names and tail entity names
        grouped_head_df = filtered_triple_ids.groupby('HeadEntityID')
        concatenated_head_data = grouped_head_df.apply(custom_concat_head).reset_index(name='Head1HopContext')
        # Merge concatenated descriptions back to triple_ids
        filtered_triple_ids_with_concat = filtered_triple_ids.merge(concatenated_head_data, on='HeadEntityID', how='left')

        concatenated_tail_data = concatenated_head_data.rename(columns={
            'HeadEntityID': 'TailEntityID',
            'Head1HopContext': 'Tail1HopContext'
        })
        filtered_triple_ids_with_concat = filtered_triple_ids_with_concat.merge(concatenated_tail_data, on='TailEntityID', how='left').fillna('')

        # Display the merged DataFrame
        print(filtered_triple_ids_with_concat.head())
        return filtered_triple_ids_with_concat

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
        relations["RelationName"] = relations["RelationName"].str[:25]
        entity_names["EntityName"] = entity_names["EntityName"].str[:40]
        entity_descriptions["EntityDescription"] = entity_descriptions["EntityDescription"].str[:200]
        triple_ids = self.filter_by_group(triple_ids, "HeadEntityID", 'RelationID', 3)
        return relations, entity_descriptions, entity_names, triple_ids

    def test_entire_wikidata_pipeline(self):
        split = "valid_tiny"
        relations, entity_ids, entity_names, entity_descriptions, triple_ids = self.read_wikidata5m_files(split)

        relations, entity_descriptions, entity_names, triple_ids = self.trim_statistically(relations, entity_descriptions, entity_names, triple_ids)

        new_df = self.prepare_fake_relation_data(relations, entity_ids, entity_names, entity_descriptions, triple_ids)

        max_length = new_df['source'].apply(len).max()
        print(f"max_length is: {max_length}")
        shuffled_df = new_df.sample(frac=1, random_state=seed).reset_index(drop=True)
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

    def test_fake_wikidata_pipeline_data_are_correct(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        relations, entity_ids, entity_names, entity_descriptions, triple_ids = self.get_fake_wikidata()
        new_df = self.prepare_fake_relation_data(relations, entity_ids, entity_names, entity_descriptions, triple_ids)

        assert new_df["EntityName_Head"].iloc[1] == "Company C"
        assert new_df["EntityName_Tail"].iloc[1] == "Company E"
        assert new_df["Head1HopContext"].iloc[1] == "<context>competes with (Company F)</context>"
        assert new_df["EntityName_Tail"].iloc[5] == "Company F"
        assert new_df["Head1HopContext"].iloc[5] == "<context>partners with (Company E)</context>"
        assert new_df["Tail1HopContext"].iloc[6] == "<context>partners with (Company E), competes with (Company F)</context>"
        assert new_df["Head1HopContext"].iloc[0] == ""
        assert new_df["Tail1HopContext"].iloc[0] == "" # because the only relation of Company D has Company A in the description
        assert new_df["Tail1HopContext"].iloc[2] == ""
        assert new_df["EntityDescription_Tail"].iloc[3].str.contains("Company A")



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
        print(tensor_restored)

    def prepare_fake_relation_data(self, relations, entity_ids, entity_names, entity_descriptions, triple_ids):

        df = self.prepare_wikidata5m_from_dfs_with_description_filter(relations, entity_descriptions, entity_names, entity_ids, triple_ids)
        df_renamed = df.rename(columns={
            'EntityName_Head': 'head',
            'EntityDescription_Head': 'head_description',
            'Head1HopContext': 'head_context',
            'RelationName': 'relation',
            'EntityName_Tail': 'tail',
            'EntityDescription_Tail': 'tail_description',
            'Tail1HopContext': 'tail_context'
        })
        # 'EntityName_Head', 'EntityDescription_Head', 'RelationID', 'RelationName', 'TailEntityID', 'QID_Tail','EntityName_Tail', 'EntityDescription_Tail'
        mem_head = df_renamed.apply(lambda x: pd.Series({"source": f"<cls>predict head: Head: <head>. Relation: {x['relation']}. Tail: {x['tail']}<tail_description>{x['tail_description']}</tail_description>",
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
        mem_tail = df_renamed.apply(lambda x: pd.Series({"source": f"<cls>predict tail: Head: {x['head']}<head_description>{x['head_description']}</head_description>Relation: {x['relation']}. Tail: <tail>",
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

        mem_relation = df_renamed.apply(lambda x: pd.Series({"source": f"<cls>predict relation: Head: {x['head']}<head_description>{x['head_description']}</head_description>Relation: <relation>. Tail: {x['tail']}<tail_description>{x['tail_description']}</tail_description>",
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

    def prepare_wikidata5m_from_dfs(self, relation_df, entity_description_df, entity_name_df, entity_id_df, triple_id_df):
        # Merge entity information
        entities = pd.merge(entity_id_df, entity_name_df, on='QID')
        entities = entities.set_index('EntityID').join(entity_description_df)
        # Enhance triple_ids with entity names and descriptions
        triple_ids = triple_id_df.merge(entities.add_suffix('_Head'), left_on='HeadEntityID', right_index=True)
        triple_ids = triple_ids.merge(entities.add_suffix('_Tail'), left_on='TailEntityID', right_index=True)
        triple_ids = triple_ids.merge(relation_df, on='RelationID')
        triple_ids = triple_ids[['HeadEntityID', 'QID_Head', 'EntityName_Head', 'EntityDescription_Head', 'RelationID', 'RelationName', 'TailEntityID', 'QID_Tail','EntityName_Tail', 'EntityDescription_Tail']]
        # Fill NaN with empty strings and convert all data to strings to avoid type issues
        columns_to_fill = ['RelationName', 'EntityName_Tail', 'EntityName_Head', 'EntityDescription_Tail',
                           'EntityDescription_Head']
        for col in columns_to_fill:
            triple_ids[col] = triple_ids[col].fillna('').astype(str)
        # Create masks for each condition
        condition1 = triple_ids['EntityName_Tail'] == triple_ids['EntityName_Head']
        # Use apply to check substring presence, row by row
        condition2 = triple_ids.apply(
            lambda row: row['EntityName_Tail'].lower() in row['EntityDescription_Head'].lower(), axis=1)
        condition3 = triple_ids.apply(
            lambda row: row['EntityName_Head'].lower() in row['EntityDescription_Tail'].lower(), axis=1)
        # Combine conditions using OR (|)
        filter_condition = condition1 | condition2 | condition3
        # Apply the filter with a negation to keep entries that do not meet any of the conditions
        filtered_triple_ids = triple_ids[~filter_condition]

        # Ensure that no concatenation results in 'nan'
        def custom_concat_head(group):
            # Apply the filters based on conditions specified in the comments
            filtered_group = group[
                ~group['EntityName_Head'].isin(group['EntityName_Tail']) &
                ~group['EntityName_Head'].isin(group['EntityDescription_Tail']) &
                ~group['EntityName_Tail'].apply(lambda x: x in group['EntityDescription_Head'].iloc[0] or x == group['EntityName_Head'].iloc[0])
                ]
            # Concatenate 'RelationName' and 'EntityName_Tail' with conditions
            return ', '.join(filtered_group['RelationName'] + ' (' + filtered_group['EntityName_Tail'] + "<tail_description>" + filtered_group['EntityDescription_Tail'] + "</tail_description>" + ')')

        def custom_concat_tail(group):
            # Apply the filters based on conditions specified in the comments
            filtered_group = group[
                ~group['EntityName_Tail'].isin(group['EntityName_Head']) &
                ~group['EntityName_Tail'].isin(group['EntityDescription_Head']) &
                ~group['EntityName_Head'].apply(lambda x: x in group['EntityDescription_Tail'].iloc[0] or x == group['EntityName_Tail'].iloc[0])
                ]
            # Concatenate 'RelationName' and 'EntityName_Tail' with conditions
            return ', '.join(filtered_group['RelationName'] + ' (' + filtered_group['EntityName_Tail'] + "<tail_description>" + filtered_group['EntityDescription_Tail'] + "</tail_description>" + ')')

        # Then, we must filter out rows where

        # Group by HeadEntityID to concatenate relation names and tail entity names
        grouped_head_df = filtered_triple_ids.groupby('HeadEntityID')
        concatenated_head_data = grouped_head_df.apply(custom_concat_head).reset_index(name='Head1HopContext')
        # Merge concatenated descriptions back to triple_ids
        triple_ids = triple_ids.merge(concatenated_head_data, on='HeadEntityID', how='left')

        grouped_tail_df = filtered_triple_ids.groupby('TailEntityID')
        concatenated_tail_data = grouped_tail_df.apply(custom_concat_tail).reset_index(name='Tail1HopContext')
        # Merge concatenated descriptions back to triple_ids
        triple_ids = triple_ids.merge(concatenated_tail_data, on='TailEntityID', how='left')

        # Display the merged DataFrame
        print(triple_ids.head())
        return triple_ids
        #triple_ids.to_parquet(f"data/wikidata5m_train_processed_{source_file}.parquet", engine='fastparquet')
