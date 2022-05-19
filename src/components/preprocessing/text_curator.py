"""TextCurator."""


import ast
import json
import logging
import os
import random
import re
from pathlib import Path
import pandas as pd
from src.components.utils.kpi_mapping import get_kpi_mapping_category

from .base_curator import BaseCurator

logger = logging.getLogger(__name__)


class TextCurator(BaseCurator):
    """TextCurator class."""

    def __init__(
        self,
        retrieve_paragraph,
        neg_pos_ratio,
        columns_to_read,
        company_to_exclude,
        seed=42,
        create_neg_samples=False,
        min_length_neg_sample=50,
        name="DataTextCurator",
        data_type="TEXT",
    ):
        """Initialize TextCurator class.

        This class is the responsible for creating ESG text dataset
        (positive and negative examples) based on the annotations.
        Args:
            retrieve_paragraph (bool): Whether or not try to extract the whole
                                       paragraph based on the annotated sentence
            neg_pos_ratio (int): Ratio between negative  to positive samples.
                                 ex. For the ratio of two, for each positive
                                 examples, approximately two negative samples
                                 will be created.
            columns_to_read (A list of str): A list of column names
            company_to_exclude (A list of str): A list of companies to exclude
            create_neg_samples (bool): Create negative samples
            min_length_neg_sample (int): minimum length of negative example
            name (str) : Name of the component
        """
        super().__init__(name)
        self.retrieve_paragraph = retrieve_paragraph
        self.neg_pos_ratio = neg_pos_ratio
        self.columns_to_read = columns_to_read
        self.company_to_exclude = company_to_exclude
        self.create_neg_samples = create_neg_samples
        self.min_length_neg_sample = min_length_neg_sample
        self.data_type = data_type
        random.seed(seed)

    def run(self, extraction_folder, annotation_excels, output_folder, kpi_df):
        """Create ESG text dataset (main method).

        Args:
            extraction_folder (str): Path to the extraction folder. In the
            extraction phase pdf are read and saved as json files.
            annotation_excels (A list of str): Paths to excel files
            output_folder (str) : Output folder to save the curated dataset.
        """
        # TODO: Move us to init
        self.extraction_folder = extraction_folder
        self.output_folder = output_folder
        self.annotation_excels = annotation_excels

        examples_list = []
        for excel_file in annotation_excels:
            examples_excel = self.process_single_annotation_file(excel_file)
            examples_list.extend(examples_excel)

        df_result = pd.DataFrame(examples_list).reset_index(drop=True)
        # Drop the unnecessary column.
        df_result.drop(["Index"], axis=1, inplace=True)

        df_result["question"] = df_result.astype({"kpi_id": "float"}, errors="ignore")[
            "kpi_id"
        ].map(get_kpi_mapping_category(kpi_df)["KPI_MAPPING"])
        # In the result csv, the following KPIs are not mapped to any questions.
        # To avoid losing any data, the following
        # KPIs should be modified manually.
        logger.warning(
            "The corresponding KPIs can not be mapped \
            to any questions and the mapped question is empty\n{}".format(
                df_result[df_result["question"].isna()]["kpi_id"].unique()
            )
        )

        # Remove the rows could not map KPI to question
        df_result = df_result[df_result.question.notnull()]
        # Remove duplicate examples
        df_result = df_result.groupby(["question", "context"]).first().reset_index()

        save_path = os.path.join(
            output_folder, "esg_{}_dataset.csv".format(self.data_type)
        )
        logger.info("Curated {} examples".format(len(df_result)))
        logger.info("Saving the dataset in {}".format(save_path))
        df_result.to_csv(save_path)

    def process_single_annotation_file(
        self, annotation_filepath, sheet_name="data_ex_in_xls"
    ):
        """Create examples for a single excel file.

        Args:
            annotation_filepath (str): Path to the annotated excel file
            sheet_name (A str): Sheet which contains data
        Returns:
            examples (list of pandas.core.series.Series): List of positive and
                                                          negative examples extracted
                                                          from excel file
        """
        logger.debug("Processing excel file {}".format(annotation_filepath))
        df = pd.read_excel(annotation_filepath, sheet_name=sheet_name)[
            self.columns_to_read
        ]

        # Filter dataframe to rows which are table and exclude certain companies
        boolean = (df.data_type == self.data_type) & (df.relevant_paragraphs.notnull())

        for exclude in self.company_to_exclude:
            boolean = boolean & (df.company != exclude)
        df = df[boolean]
        df["annotator"] = os.path.basename(annotation_filepath)

        examples = []
        for i, row in df.iterrows():
            row["Index"] = i
            positive_examples = self.create_pos_examples(row.copy())

            examples.extend(positive_examples)

            if self.create_neg_samples:
                negative_examples = self.create_negative_examples(row.copy())
                if negative_examples is not None:
                    examples.extend(negative_examples)

        return examples

    def create_pos_examples(self, row):
        """Create pos examples (Given each row of annotations).

        Note: this method can extract the whole paragraph in the pdf where the
        relevant_paragraph is mentioned.
        For each row, there might be more than one positive examples if the
        list of relevant_paragraph column contains more than one element.

        Args:
            row (pandas.core.series.Series): each row of pandas data frame
        Returns:
            pos_rows (list of pandas.core.series.Series): A list of positive examples,
                                                          each row has two more members,
                                                          "context" and "label" of 1.
        """
        # Change the format of relevant_paragraphs from string to list
        sentences = self.process_relevant_sentences(row)
        # The above method will return None, if it cannot process
        # relevance_paragraph column.
        if sentences is None:
            return []

        if self.retrieve_paragraph and not pd.isnull(row["source_page"]):
            paragraphs = self.get_full_paragraph(row, sentences)
            # If method cannot extract relevant paragraphs based on relevant
            # sentences, it will return empty list.
            if len(paragraphs) == 0:
                paragraphs = sentences
        else:
            paragraphs = sentences

        pos_rows = []
        for p in paragraphs:
            row_copy = row.copy()
            row_copy["context"] = p
            row_copy["label"] = 1
            pos_rows.append(row_copy)

        return pos_rows

    def create_negative_examples(self, row):
        """Create negative examples for each row.

        To achieve this:
        - If the source pdf is presented and extracted, we choose a random page,
          except source page and choose a random paragraph within that.
        - If the extracted pdf is not available, we look for the a random
          extracted pdf and choose a random paragraph inside that.

        Args:
            row (pandas.core.series.Series): each row of pandas data frame
        Return:
            neg_rows (list of pandas.core.series.Series): A list of negative
                                                          examples, each row has
                                                          two more members,
                                                          context and label of 0.
        """
        pdf_content = self.load_pdf_content(row)
        #  if the corresponding pdf to a row is not presented, a random pdf is
        # picked to create negative example
        if len(pdf_content) == 0:
            random_json_path = random.choice(
                list(Path(self.extraction_folder).rglob("*.json"))
            )
            with open(os.path.join(self.extraction_folder, random_json_path)) as f:
                pdf_content = [json.load(f)]

        try:
            selected_pages = [p - 1 for p in ast.literal_eval(row["source_page"])]
        except SyntaxError:
            if len(self.load_pdf_content(row)) == 0:
                selected_pages = []
            else:
                return None

        neg_rows = []
        for _ in range(int(self.neg_pos_ratio)):
            while True:
                if (len(pdf_content[0]) - 1) < 3:
                    return None
                negative_page = random.randint(3, len(pdf_content[0]) - 1)
                if negative_page in selected_pages:
                    continue
                negative_page_content = pdf_content[0][str(negative_page)]
                if len(negative_page_content) == 0:
                    continue
                negative_context = random.choice(negative_page_content)
                negative_context = self.clean_text(negative_context)
                if len(negative_context) < self.min_length_neg_sample:
                    continue
                break

            # Before assigning value, create a copy of it
            row_copy = row.copy()
            row_copy["context"] = negative_context
            row_copy["label"] = 0
            neg_rows.append(row_copy)

        return neg_rows

    def process_relevant_sentences(self, row):
        """Extract relevant paragraph based on the 'relevant_paragraphs' column.

        This method will check the format of relevant paragraph and if it does
        not follow the standard, the logger will print out information about that
        and will return None.
        Note: To not lose any data, the excel files can be manually modified and
        the program should be run again.

        Args:
            row (pandas.core.series.Series)
        Return:
            sentence_revised (list of str) List of relevant sentences
        """
        sentence_revised = self.clean_text(row["relevant_paragraphs"])

        if sentence_revised.startswith("[") or sentence_revised.endswith("]"):
            try:
                return ast.literal_eval(sentence_revised)
            except SyntaxError:
                # This happens if there is an issue in the relevant paragraph.
                # + 1 special case
                # TODO: Solve the especial case: it happens if there is double
                # quotes in the sentence.
                # 2 is added because to be compatible, so it would be compatible
                # with what you see in MS excel.
                logger.warning(
                    "Could not process row number {} in {}".format(
                        (row["Index"] + 2), row["annotator"]
                    )
                )
                return None
        else:
            # To support cases where relevant paragraph are given as strings.
            logger.info(
                "Not in a list format row number {} , {}".format(
                    (row["Index"] + 2), row["annotator"]
                )
            )
            return [sentence_revised]

    def get_full_paragraph(self, row, relevant_sentences):
        r"""Find the full paragraph where the relevant_sentence column is coming from.

        To achieve this:

        The content of the pdf mentioned in the source_page is loaded.
        The content of page mentioned in source_page is selected.
        The extracted text is divided by paragraphs "\n\n"
        For each paragraphs in the page, will find a match for relevant_paragraph.
        Note: The result can be an empty list if the paragraph can not retrieved.

        Args:
            row (pandas.core.series.Series): Each row of pandas dataframe.
            relevant_sentences (list of str): List of processed relevant_paragraphs.
        Returns:
            matches_list (list of str): list of full paragraphs.
        """
        pdf_content = self.load_pdf_content(row)
        try:
            source_page = ast.literal_eval(row["source_page"])
        except SyntaxError:
            logger.info(
                "Can not process source page in row {} of {} ".format(
                    (row["Index"] + 2), row["annotator"]
                )
            )
            return []
        # pdfminer starts the page counter as 0 while for pdf viewers the first
        # page is numbered as 1.
        selected_pages = [p - 1 for p in source_page]
        paragraphs = [
            pdf.get(str(p), []) for p in selected_pages for pdf in pdf_content
        ]
        paragraphs_flat = [item for sublist in paragraphs for item in sublist]
        matches_list = []
        for pattern in relevant_sentences:
            special_regex_char = [
                "(",
                ")",
                "^",
                "+",
                "*",
                "$",
                "|",
                "\\",
                "?",
                "[",
                "]",
                "{",
                "}",
            ]
            # If the sentences contain the especial character we should put \
            # before them for literal match.
            pattern = "".join(
                ["\\" + c if c in special_regex_char else c for c in pattern]
            )
            for single_par in paragraphs_flat:
                single_par_clean = self.clean_text(single_par)
                match = re.search(pattern, single_par_clean, re.I)
                if match is not None:
                    matches_list.append(single_par_clean)
                    break

        return matches_list

    def load_pdf_content(self, row):
        """Load the content of a pdf file.

        If the extraction step is passed, the json file should be in the
        extraction_folder.
        Args:
            row (list of pandas.core.series.Series)
        Returns:
                (list of dict): List of pdfs' content that has the relevant name
                                after extraction.
        """
        # The naming format is used in extraction phase.
        extracted_filename = (
            os.path.splitext(str(row["source_file"]))[0] + "-" + str(row["company"])
        )
        # Get all the files in extraction folder that has the desired name
        extracted_paths = [
            path
            for path in os.listdir(self.extraction_folder)
            if extracted_filename in path
        ]

        pdf_contents = []
        for path in extracted_paths:
            with open(os.path.join(self.extraction_folder, path)) as f:
                pdf_contents.append(json.load(f))
        return pdf_contents
        # TODO: Support cases where the source pdf exists in the pdf_folder but it is not extracted.
        # else:
        #     # Look for the pdf name in the pdf folders that are the same as source column.
        #     source_pdf_path = self.get_path_pdf(config.PDF_FOLDER, row.source_file)
        #     if source_pdf_path is None:
        #         logger.info("The related mentioned pdf {} in {} not found, and the extracted not found"
        #         .format(row.annotator, row.source))
        #         return None
        #     # Get the content for all files named as source file name in annotations.
        #     pdf_content = [PDFTextExtractor.extract_pdf_by_page(path) for path in source_pdf_path]
        # return pdf_content
