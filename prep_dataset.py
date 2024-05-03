import asyncio
import random
import re
import unittest
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
import pandas as pd
import torch
from cassandra.concurrent import execute_concurrent_with_args
from tqdm import tqdm
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
1\t4\t2

                """)
# 4\t1\t3

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
        print("DF filtered logically")

        return filtered_triple_ids.reset_index(drop=True)

    def build_triples_with_descriptions(self, entity_description_df, entity_id_df, entity_name_df, relation_df, triple_id_df):
        # Merge entity information
        print("Merging dataframes from source files")
        entities = pd.merge(entity_id_df, entity_name_df, on='QID')
        entities = entities.set_index('EntityID').join(entity_description_df)
        # Enhance triple_ids with entity names and descriptions
        triple_ids = triple_id_df.merge(entities.add_suffix('_Head'), left_on='HeadEntityID', right_index=True)
        triple_ids = triple_ids.merge(entities.add_suffix('_Tail'), left_on='TailEntityID', right_index=True)
        triple_ids = triple_ids.merge(relation_df, on='RelationID')
        triple_ids = triple_ids[
            ['HeadEntityID', 'QID_Head','TailEntityID', 'QID_Tail',  'RelationID', 'EntityName_Head', 'EntityDescription_Head', 'RelationName',
              'EntityName_Tail', 'EntityDescription_Tail']]
        # Fill NaN with empty strings and convert data to strings to avoid type issues
        columns_to_fill = ['RelationName', 'EntityName_Tail', 'EntityName_Head', 'EntityDescription_Tail',
                           'EntityDescription_Head']
        for col in columns_to_fill:
            triple_ids[col] = triple_ids[col].fillna('').astype(str)
        print("built df of quintuples")
        return triple_ids
    def trim_statistically_and_truncate(self, relations, entity_descriptions, entity_names, triple_ids):
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
        new_df = pd.concat([mem_head, mem_tail, mem_relation]).drop_duplicates().reset_index(drop=True)
        return new_df

    def test_build_all_wikidata_splits(self):
        print("Starting validation tiny data prep.")
        self.build_entire_pipeline_real("valid_tiny")
        print("Starting validation data prep.")
        self.build_entire_pipeline_real("valid")
        print("Starting test data prep.")
        self.build_entire_pipeline_real("test")
        print("Starting training data prep.")
        #self.build_entire_wikidata_pipeline_real("train")

    def build_graph_dfs_from_split(self, split: str):
        relation_df, entity_id_df, entity_name_df, entity_description_df, triple_id_df = self.read_wikidata5m_files(split)
        relation_df, entity_description_df, entity_name_df, triple_id_df = self.trim_statistically_and_truncate(relation_df, entity_description_df, entity_name_df, triple_id_df)
        triple_df = self.build_triples_with_descriptions(entity_description_df, entity_id_df, entity_name_df, relation_df,
                                                         triple_id_df)
        prefiltered_df = self.filter_triples_logically_before_join(triple_df)
        graph = self.build_networkx_graph(prefiltered_df)
        results_df, results_annotated_df = self.filter_adjacent_nodes(graph)
        # Order the columns to make dataframe easier to work with:
        results_annotated_df = results_annotated_df[['head', 'head_description', 'relation', 'tail', 'tail_description', 'head_context', 'tail_context', 'source', 'target', 'objective']]
        print("returning from build_graph_dfs_from_split(..)")
        return results_df, results_annotated_df
    def test_inspect_source_length_of_training_set(self):
        results_df, results_annotated_df = self.build_graph_dfs_from_split("valid_tiny")
        # Calculate the length of each string in the 'source' column
        # Calculate the length of each string in the 'source' and 'target' columns
        print("Computing lengths")
        source_lengths = results_annotated_df['source'].str.len()
        print("Computing max length of source")
        max_length = results_annotated_df['source'].apply(len).max() # need to replace with truncation step?
        print(f"max_length is: {max_length}")
        target_lengths = results_annotated_df['target'].str.len()

        # Set up the figure with two subplots
        plt.figure(figsize=(12, 6))

        # Histogram for the 'source' column
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        plt.hist(source_lengths, bins=30, color='blue', edgecolor='black')
        plt.title('Histogram of Source Lengths')
        plt.xlabel('Length of Source')
        plt.ylabel('Frequency')
        plt.grid(True)

        # Histogram for the 'target' column
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        plt.hist(target_lengths, bins=30, color='green', edgecolor='black')
        plt.title('Histogram of Target Lengths')
        plt.xlabel('Length of Target')
        plt.grid(True)

        # Display the plots
        plt.tight_layout()  # Adjusts subplot params so that the subplot(s) fits in to the figure area.
        plt.show()

    def get_tokenizer(self):
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        special_tokens = [
            "<cls>", "<head>", "<head_description>", "</head_description>", "<mask>","<relation>", "<tail>", "<tail_description>","</tail_description>","<relation>", "<end>"
        ]
        tokenizer.add_tokens(special_tokens)
        return tokenizer

    def build_entire_pipeline_real(self, split: str):
        print(f"Processing {split}")
        results_df, results_annotated_df = self.build_graph_dfs_from_split(split)
        print("Computing source length")
        max_length_source = results_annotated_df['source'].apply(len).max() # need to replace with truncation step?
        print(f"max length of source text is: {max_length_source}")
        max_length_target = results_annotated_df['target'].apply(len).max()
        print(f"max length of target text is: {max_length_target}")
        shuffled_df = results_annotated_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"Writing file")
        shuffled_df.to_parquet(f"data/wikidata5m_train_processed_filtered_{split}.parquet", engine='fastparquet')
        print(f"Done processing {split}")

    def test_graph_building(self):

        relation_df, entity_id_df, entity_name_df, entity_description_df, triple_id_df = self.get_fake_wikidata()
        #relation_df, entity_description_df, entity_name_df, triple_id_df = self.trim_statistically_and_truncate(relation_df, entity_description_df, entity_name_df, triple_id_df)
        triple_df = self.build_triples_with_descriptions(entity_description_df, entity_id_df, entity_name_df, relation_df,
                                                         triple_id_df)
        triple_df = triple_df.sort_values(by=['EntityName_Head', 'EntityName_Tail'])
        prefiltered_df = self.filter_triples_logically_before_join(triple_df)

        graph = self.build_networkx_graph(prefiltered_df)
        results_df, results_annotated_df = self.filter_adjacent_nodes(graph)
        results_df = results_df[['head', 'head_description', 'relation', 'tail', 'tail_description', 'head_context', 'tail_context']].sort_values(by=['head', 'relation', 'tail'])

        assert results_df.iloc[0]['head'] == 'Company A'
        assert results_df.iloc[0]['head_description'] == 'Tech firm'
        assert results_df.iloc[0]['relation'] == 'merged with'
        assert results_df.iloc[0]['tail'] == 'Company B'
        assert results_df.iloc[0]['tail_description'] == 'Startup that works with Company C. Similar to company F'
        assert results_df.iloc[0]['head_context'] == '<context>partners with (Company D), merged with (Company D)</context>'
        assert results_df.iloc[0]['tail_context'] == ''

        assert results_df.iloc[1]['head'] == 'Company A'
        assert results_df.iloc[1]['head_description'] == 'Tech firm'
        assert results_df.iloc[1]['relation'] == 'merged with'
        assert results_df.iloc[1]['tail'] == 'Company D'
        assert results_df.iloc[1]['tail_description'] == 'Manufacturer'
        assert results_df.iloc[1]['head_context'] == '<context>merged with (Company B)</context>'
        assert results_df.iloc[1]['tail_context'] == ''

        assert results_df.iloc[2]['head'] == 'Company A'
        assert results_df.iloc[2]['head_description'] == 'Tech firm'
        assert results_df.iloc[2]['relation'] == 'partners with'
        assert results_df.iloc[2]['tail'] == 'Company D'
        assert results_df.iloc[2]['tail_description'] == 'Manufacturer'
        assert results_df.iloc[2]['head_context'] == '<context>merged with (Company B)</context>'
        assert results_df.iloc[2]['tail_context'] == ''

        assert results_df.iloc[3]['head'] == 'Company B'
        assert results_df.iloc[3]['head_description'] == 'Startup that works with Company C. Similar to company F'
        assert results_df.iloc[3]['relation'] == 'acquired by'
        assert results_df.iloc[3]['tail'] == 'Company E'
        assert results_df.iloc[3]['tail_description'] == 'E-commerce company, a company that worked with Company A'
        assert results_df.iloc[3]['head_context'] == ''
        assert results_df.iloc[3]['tail_context'] == ''

        assert results_df.iloc[4]['head'] == 'Company C'
        assert results_df.iloc[4]['head_description'] == 'Retailer'
        assert results_df.iloc[4]['relation'] == 'competes with'
        assert results_df.iloc[4]['tail'] == 'Company D'
        assert results_df.iloc[4]['tail_description'] == 'Manufacturer'
        assert results_df.iloc[4]['head_context'] == '<context>partners with (Company E), competes with (Company F)</context>'
        assert results_df.iloc[4]['tail_context'] == '<context>acquired by (Company F)</context>'

        assert results_df.iloc[5]['head'] == 'Company C'
        assert results_df.iloc[5]['head_description'] == 'Retailer'
        assert results_df.iloc[5]['relation'] == 'competes with'
        assert results_df.iloc[5]['tail'] == 'Company F'
        assert results_df.iloc[5]['tail_description'] == 'Tech firm, the parent of Company A'
        assert results_df.iloc[5]['head_context'] == '<context>competes with (Company D), partners with (Company E)</context>'
        assert results_df.iloc[5]['tail_context'] == ''

        assert results_df.iloc[6]['head'] == 'Company C'
        assert results_df.iloc[6]['head_description'] == 'Retailer'
        assert results_df.iloc[6]['relation'] == 'partners with'
        assert results_df.iloc[6]['tail'] == 'Company E'
        assert results_df.iloc[6]['tail_description'] == 'E-commerce company, a company that worked with Company A'
        assert results_df.iloc[6]['head_context'] == '<context>competes with (Company D), competes with (Company F)</context>'
        assert results_df.iloc[6]['tail_context'] == ''

        assert results_df.iloc[7]['head'] == 'Company D'
        assert results_df.iloc[7]['head_description'] == 'Manufacturer'
        assert results_df.iloc[7]['relation'] == 'acquired by'
        assert results_df.iloc[7]['tail'] == 'Company F'
        assert results_df.iloc[7]['tail_description'] == 'Tech firm, the parent of Company A'
        assert results_df.iloc[7]['head_context'] == ''
        assert results_df.iloc[7]['tail_context'] == '<context>sold products to (Company C)</context>'

        assert results_df.iloc[8]['head'] == 'Company F'
        assert results_df.iloc[8]['head_description'] == 'Tech firm, the parent of Company A'
        assert results_df.iloc[8]['relation'] == 'sold products to'
        assert results_df.iloc[8]['tail'] == 'Company C'
        assert results_df.iloc[8]['tail_description'] == 'Retailer'
        assert results_df.iloc[8]['head_context'] == ''
        assert results_df.iloc[8]['tail_context'] == '<context>competes with (Company D), partners with (Company E)</context>'
        print(results_df.head())

        assert results_annotated_df.iloc[0]['head'] == 'Company A'
        assert results_annotated_df.iloc[0]['head_description'] == 'Tech firm'
        assert results_annotated_df.iloc[0]['relation'] == 'merged with'
        assert results_annotated_df.iloc[0]['tail'] == 'Company B'
        assert results_annotated_df.iloc[0]['tail_description'] == 'Startup that works with Company C. Similar to company F'
        assert results_annotated_df.iloc[0]['head_context'] == '<context>partners with (Company D), merged with (Company D)</context>'
        assert results_annotated_df.iloc[0]['tail_context'] == ''
        assert results_annotated_df.iloc[0]['source'] == '<cls>predict head: Head: <head>. Relation: merged with. Tail: Company B<tail_description>Startup that works with Company C. Similar to company F</tail_description>'
        assert results_annotated_df.iloc[0]['target'] == '<head>Company A<end>'
        assert results_annotated_df.iloc[0]['objective'] == 'predict_head'

        assert results_annotated_df.iloc[1]['head'] == 'Company A'
        assert results_annotated_df.iloc[1]['head_description'] == 'Tech firm'
        assert results_annotated_df.iloc[1]['relation'] == 'merged with'
        assert results_annotated_df.iloc[1]['tail'] == 'Company B'
        assert results_annotated_df.iloc[1]['tail_description'] == 'Startup that works with Company C. Similar to company F'
        assert results_annotated_df.iloc[1]['head_context'] == '<context>partners with (Company D), merged with (Company D)</context>'
        assert results_annotated_df.iloc[1]['tail_context'] == ''
        assert results_annotated_df.iloc[1]['source'] == '<cls>predict tail: Head: Company A<head_description>Tech firm</head_description>Relation: merged with. Tail: <tail>'
        assert results_annotated_df.iloc[1]['target'] == '<tail>Company B<end>'
        assert results_annotated_df.iloc[1]['objective'] == 'predict_tail'

        assert results_annotated_df.iloc[2]['head'] == 'Company A'
        assert results_annotated_df.iloc[2]['head_description'] == 'Tech firm'
        assert results_annotated_df.iloc[2]['relation'] == 'merged with'
        assert results_annotated_df.iloc[2]['tail'] == 'Company B'
        assert results_annotated_df.iloc[2]['tail_description'] == 'Startup that works with Company C. Similar to company F'
        assert results_annotated_df.iloc[2]['head_context'] == '<context>partners with (Company D), merged with (Company D)</context>'
        assert results_annotated_df.iloc[2]['tail_context'] == ''
        assert results_annotated_df.iloc[2]['source'] == '<cls>predict relation: Head: Company A<head_description>Tech firm</head_description>Relation: <relation>. Tail: Company B<tail_description>Startup that works with Company C. Similar to company F</tail_description>'
        assert results_annotated_df.iloc[2]['target'] == '<relation>merged with<end>'
        assert results_annotated_df.iloc[2]['objective'] == 'predict_relation'

    def build_networkx_graph(self, dataframe):
        # Create the graph
        G = nx.from_pandas_edgelist(
            dataframe,
            source='HeadEntityID',
            target='TailEntityID',
            edge_attr='RelationName',
            create_using=nx.MultiDiGraph()
        )
        print("Built graph from pandas edge list. Adding node attributes next.")
        # Add node attributes for head entities
        # Prepare updates for head entities
        head_updates = {
            row['HeadEntityID']: {
                'EntityName': row['EntityName_Head'],
                'EntityDescription': row['EntityDescription_Head']
            } for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Preparing head entities")
        }

        # Prepare updates for tail entities
        tail_updates = {
            row['TailEntityID']: {
                'EntityName': row['EntityName_Tail'],
                'EntityDescription': row['EntityDescription_Tail']
            } for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Preparing tail entities")
        }

        # Apply updates to the graph
        for node_id, attributes in tqdm(head_updates.items(), desc="Updating head nodes"):
            G.nodes[node_id].update(attributes)

        for node_id, attributes in tqdm(tail_updates.items(), desc="Updating tail nodes"):
            G.nodes[node_id].update(attributes)
        return G

    def build_node_context(self, G, source_node_id, target_node_id):
        head_data = G.nodes[source_node_id]
        tail_data = G.nodes[target_node_id]
        head_name_and_description = f"{head_data['EntityName']} {head_data['EntityDescription']}"
        tail_name_and_description = f"{tail_data['EntityName']} {tail_data['EntityDescription']}"
        adjacents = []
        for adjacent_node_id in G.successors(source_node_id):
            adjacent_node = G.nodes[adjacent_node_id]
            adjacent_name_and_description = f"{adjacent_node['EntityName']} {adjacent_node['EntityDescription']}"

            adjacent_name_in_head = adjacent_node["EntityName"] in head_name_and_description
            adjacent_name_in_tail = adjacent_node["EntityName"] in tail_name_and_description
            head_name_in_adjacent = head_data['EntityName'] in adjacent_name_and_description
            tail_name_in_adjacent = tail_data['EntityName'] in adjacent_name_and_description

            any_true = any([adjacent_name_in_head, adjacent_name_in_tail, head_name_in_adjacent, tail_name_in_adjacent])

            if not any_true:
                relations = [attrs['RelationName'] for attrs in G.get_edge_data(source_node_id, adjacent_node_id).values()]
                for relation in relations:
                    if relation is not None:
                        # Add relation with name as context
                        successor_context = f"{relation} ({adjacent_node['EntityName']})"
                        adjacents.append(successor_context)
                    else:
                        print(f"WARNING: Missing relation between nodes when building graph: {head_data['EntityName']} and {tail_data['EntityName']}")
        # join adjacents into csv
        adjacents_text = ", ".join(adjacents)
        head_context = "<context>" + adjacents_text + "</context>" if len(adjacents_text) > 0 else ''
        return head_context
    def filter_adjacent_nodes(self, G):
        results_annotated = []
        results = []
        # Iterate over each pair of nodes
        for H, T in tqdm(G.edges(), total=G.number_of_edges(), desc="Processing edges"):
            head_data = G.nodes[H]
            tail_data = G.nodes[T]

            # Find adjacent nodes for H
            head_context = self.build_node_context(G, H, T)

            # Find adjacent nodes for T
            tail_context = self.build_node_context(G, T, H)

            # Collect the relationship type
            relations = [attrs['RelationName'] for attrs in G.get_edge_data(H, T).values()]

            for relation in relations:
                if relation is None:
                    print(f"WARNING: Missing relation between nodes when traversing graph: {head_data['EntityName']} and {tail_data['EntityName']}")
                else:
                    # Predict head objective:
                    results.append({
                        'head': head_data['EntityName'],
                        'head_description': head_data['EntityDescription'],
                        'head_context' : head_context,
                        'relation': relation,
                        'tail': tail_data['EntityName'],
                        'tail_description': tail_data['EntityDescription'],
                        'tail_context': tail_context
                    })

                    results_annotated.append({
                        'source': f"<cls>predict head: Head: <head>. Relation: {relation}. Tail: {tail_data['EntityName']}<tail_description>{tail_data['EntityDescription']}</tail_description>",
                        "target": f"<head>{head_data['EntityName']}<end>",
                        'head': head_data['EntityName'],
                        'head_description': head_data['EntityDescription'],
                        'head_context' : head_context,
                        'relation': relation,
                        'tail': tail_data['EntityName'],
                        'tail_description': tail_data['EntityDescription'],
                        'tail_context': tail_context,
                        "objective": "predict_head"
                    })

                    # Predict tail objective:
                    results_annotated.append({
                        'source': f"<cls>predict tail: Head: {head_data['EntityName']}<head_description>{head_data['EntityDescription']}</head_description>Relation: {relation}. Tail: <tail>",
                        "target": f"<tail>{tail_data['EntityName']}<end>",
                        'head': head_data['EntityName'],
                        'head_description': head_data['EntityDescription'],
                        'head_context' : head_context,
                        'relation': relation,
                        'tail': tail_data['EntityName'],
                        'tail_description': tail_data['EntityDescription'],
                        'tail_context': tail_context,
                        "objective": "predict_tail"
                    })

                    # Predict relation objective:

                    results_annotated.append({
                        'source': f"<cls>predict relation: Head: {head_data['EntityName']}<head_description>{head_data['EntityDescription']}</head_description>Relation: <relation>. Tail: {tail_data['EntityName']}<tail_description>{tail_data['EntityDescription']}</tail_description>",
                        "target": f"<relation>{relation}<end>",
                        'head': head_data['EntityName'],
                        'head_description': head_data['EntityDescription'],
                        'head_context' : head_context,
                        'relation': relation,
                        'tail': tail_data['EntityName'],
                        'tail_description': tail_data['EntityDescription'],
                        'tail_context': tail_context,
                        "objective": "predict_relation"
                    })
        results_annotated_df = pd.DataFrame(results_annotated).drop_duplicates().reset_index(drop=True)
        results_df = pd.DataFrame(results).drop_duplicates().reset_index(drop=True)
        print("Dropped duplicates")
        return results_df, results_annotated_df
