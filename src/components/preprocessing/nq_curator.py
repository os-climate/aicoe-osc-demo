import logging
import os
from ast import literal_eval
from glob import glob
from multiprocessing import Pool, cpu_count

import pandas as pd
import tqdm

from src.components.utils.nq_utils import (
    get_text_section,
    remove_html_tags,
    contains_table,
    is_not_short,
)
from .base_component import BaseComponent

logger = logging.getLogger(__name__)


class NQCurator(BaseComponent):
    """Component that creates balanced datasets for text and table data from the extracted NQ data
    Args:
        input_dir(str): The directory where the extracted CSVs are located.
        output_dir(str): The directory where the curated CSVs will be saved.
        extract_text(bool): Set to True to curate the table data
        extract_tables(bool): Set to True to curate the text data
        negative_from_other_docs(bool): If set to True, the negative examples are sampled from paragraphs from
         the same document from which the positive examples are taken from. If False, they will be sampled from
         other documents.

    """

    def __init__(
        self,
        input_dir,
        output_dir,
        extract_text,
        extract_tables,
        negative_from_other_docs=True,
        name="NQCurator",
    ):
        super().__init__(name)
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._extract_text = extract_text
        self._extract_tables = extract_tables
        self._negative_from_other_docs = negative_from_other_docs

    @staticmethod
    def _extract_paragraphs(df):
        """Extracts the content of the relevant and candidate answer paragraphs from the token start and end indices
        Args:
            df(Pandas dataframe): The input dataframe.
        Returns:
            df(Pandas dataframe): Same dataframe as the input, with two additional columns: `document_text` and
                `other_long_answer_candidates`
        """
        logger.info("Extracting relevant paragraphs...")
        df["relevant_excerpt"] = [
            get_text_section(doc_tokens)
            for doc_tokens in tqdm.tqdm(
                zip(
                    df["document_text"].tolist(),
                    df["long_answer_start"].tolist(),
                    df["long_answer_end"].tolist(),
                ),
                total=len(df),
            )
        ]

        logger.info("Extracting other long answer candidates...")
        process_count = cpu_count() - 1
        with Pool(process_count) as p:
            all_other_candidates = [
                list(
                    p.map(
                        get_text_section,
                        [(doc, p["start_token"], p["end_token"]) for p in paras],
                    )
                )
                for (doc, paras) in tqdm.tqdm(
                    zip(
                        df["document_text"].tolist(),
                        df["other_long_answer_candidates"].tolist(),
                    ),
                    total=len(df),
                )
            ]

        df["other_candidates"] = all_other_candidates

        return df

    @staticmethod
    def create_relevance_dataset(input_df):
        """
        Process NQ dataset to get both a text dataset and a table dataset to train a
         relevance classifier
        Args:
             input_df (dataframe): original NQ dataset loaded using jsonl_to_df
        Returns:
            df_text (dataframe): A dataframe containing the questions, positive and negative paragraph examples,
             and the labels
            df_table (dataframe): A dataframe containing the questions, positive and negative table examples,
             and labels
        """

        # Create dataframe with only question, text and label
        logger.info("Creating relevance dataframe")
        all_data = [
            {"question": q, "text": t, "label": 1}
            for (q, t) in zip(input_df["question_text"], input_df["relevant_excerpt"])
        ]
        all_data += [
            {"question": q, "text": t, "label": 0}
            for (q, texts) in zip(
                input_df["question_text"], input_df["other_candidates"]
            )
            for t in texts
        ]

        del input_df
        df_relevance = pd.DataFrame(all_data)

        df_relevance["with_table"] = df_relevance["text"].map(contains_table)

        # Changed df_relevance[(df_relevance["with_table"] == True)] to this
        df_table = df_relevance[df_relevance["with_table"]].copy()
        df_text = df_relevance.drop(df_table.index).reset_index(drop=True).copy()
        df_table = df_table.reset_index(drop=True)

        df_table.drop(columns="with_table", inplace=True)
        df_text.drop(columns="with_table", inplace=True)

        # Remove HTML tags
        logger.info("Removing HTML tags...")
        df_text["text"] = df_text["text"].map(remove_html_tags)

        # Remove too short paragraphs
        logger.info("Removing short paragraphs...")
        df_text = df_text[df_text["text"].map(is_not_short)]

        df_text = df_text.reset_index(drop=True)

        # Keep only questions that have both positive and negative samples after
        # filtering
        groups = df_text.groupby("question")
        df_text = groups.filter(
            lambda x: (1 in list(x["label"])) & (0 in list(x["label"]))
        )

        return df_text, df_table

    def _build_balanced_dataset(self, df):
        """
        Select negative samples to build a balanced NQ relevance dataset and save it
        to csv
        Args:
        df (dataframe): Processed NQ data with text only or table only data
        (output of to_relevance_dataset() method)
        Returns:
            balanced_dataset (dataframe)
        """

        # Get positive samples
        positive_samples = df.loc[df.label == 1].reset_index(drop=True)
        # Get negative candidates
        negative_candidates = df.loc[df.label == 0].reset_index(drop=True)
        # Shuffle all the candidates
        negative_candidates = negative_candidates.sample(
            frac=1, random_state=42
        ).reset_index(drop=True)
        # Select one negative candidate per question
        negative_samples = negative_candidates.groupby("question").first().reset_index()

        if self._negative_from_other_docs:
            # Shuffle the questions
            new_questions = (
                negative_samples["question"]
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )
            # Assign new questions to the selected negative candidates
            negative_samples["question"] = new_questions

        balanced_dataset = (
            pd.concat([positive_samples, negative_samples])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        return balanced_dataset

    def _run(self, df_path):
        """The run method for a single chunk of data. Will curate a text and a table dataset from the `df_path`
        Args:
            df_path(str): Path to the CSV where the chunk of parsed data is saved at
        Returns:
            df_text_balanced(Pandas df): A dataframe with equal number of pos and neg text examples
            df_table_balanced(Pandas df): A dataframe with equal number of pos and neg table examples
        """
        df = pd.read_csv(
            df_path, converters={"other_long_answer_candidates": literal_eval}
        )
        df = self._extract_paragraphs(df)
        df_text, df_table = self.create_relevance_dataset(df)

        df_text_balanced = (
            self._build_balanced_dataset(df_text) if self._extract_text else None
        )
        df_table_balanced = (
            self._build_balanced_dataset(df_table) if self._extract_tables else None
        )

        return df_text_balanced, df_table_balanced

    def run(self):
        """Loads all the extracted CSVs, and processes them one by one, curating a text and a table dataset from each
        The datasets are then aggregated and saved as CSV.
        """
        logger.info("Running the Curation stage")
        logger.info("=" * 30)
        processed_files_paths = sorted(glob(os.path.join(self._input_dir, "*.csv")))
        list_df_text_balanced = []
        list_df_table_balanced = []
        for path in processed_files_paths:
            df_text_balanced, df_table_balanced = self._run(path)
            list_df_text_balanced.append(df_text_balanced)
            list_df_table_balanced.append(df_table_balanced)

        if self._extract_text:
            df_text_agg = pd.concat(list_df_text_balanced)
            df_text_agg.to_csv(
                os.path.join(self._output_dir, "NQ_text_relevance_balanced.csv")
            )
        if self._extract_tables:
            df_table_agg = pd.concat(list_df_table_balanced)
            df_table_agg.to_csv(
                os.path.join(self._output_dir, "NQ_table_relevance_balanced.csv")
            )
