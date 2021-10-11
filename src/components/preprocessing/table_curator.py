import logging
import os
import random
import numpy as np
import xlrd
from collections import defaultdict

import pandas as pd
from fuzzywuzzy import fuzz

from src.components.utils.kpi_mapping import KPI_MAPPING, KPI_CATEGORY
from .base_curator import BaseCurator

logger = logging.getLogger(__name__)


class TableCurator(BaseCurator):
    def __init__(
        self,
        neg_pos_ratio,
        columns_to_read,
        company_to_exclude,
        seed=42,
        create_neg_samples=False,
        name="DataTableCurator",
        data_type="TABLE",
    ):
        """This class is the responsible for creating ESG table dataset
            (positive and negative examples) based on the annotations.
        Args:
            neg_pos_ratio (int): Ratio between negative to positive samples.
                                 ex. For the ratio of two, for each positive examples,
                                 approximately two negative samples will be created.
            create_neg_samples (bool): Create negative samples
            columns_to_read (A list of str): Columns to read from excels
            company_to_exclude (A list of str): Companies to exclude
            name (str) : Name of the component
        """
        super().__init__(name)
        self.neg_pos_ratio = neg_pos_ratio
        self.create_neg_samples = create_neg_samples
        self.columns_to_read = columns_to_read
        self.data_type = data_type
        self.company_to_exclude = company_to_exclude
        random.seed(seed)

    def run(self, extraction_folder, annotation_excels, output_folder):
        """This is the main method for creating ESG table dataset. It saves
            all examples in a csv.

        Args:
            extraction_folder (str): Path to the extraction folder. In the
                extraction phase pdf are read and saved as csv files. Contains a
                table_meta.json file
            annotation_excels (A list of str): Paths to annotation excels
            output_folder (str) : Output folder to save the curated dataset.
        """
        self.extraction_folder = extraction_folder
        self.output_folder = output_folder
        self.annotation_excels = annotation_excels
        self.filename_to_stringarr = self.__obtain_filename_to_strarr()

        examples_list = []
        for excel_file in self.annotation_excels:
            examples_excel = self.process_single_annotation_file(excel_file)
            examples_list.extend(examples_excel)

        df_result = pd.DataFrame(examples_list).reset_index(drop=True)
        df_result.columns = [
            "Company",
            "Year",
            "Question",
            "Answer",
            "Table_filename",
            "Label",
        ]

        # TODO: Remove hard coded
        df_result.to_csv(
            os.path.join(
                self.output_folder, "esg_{}_dataset.csv".format(self.data_type)
            )
        )

    def create_pos_examples(self, row, tables_samepdf):
        """
        Args:
            row (A pandas.Series)
            tables_samepdf (A dict): Key= Page number (A str), value = A list of
                                        table csv paths

        Returns:
            multi_examples (A list of lists): [[str, str, str, str, str, int]].
                                             [] if no table is found in the page
        """
        company = row["company"]
        year = row["year"]
        answer = row["answer"]
        source_page = row["source_page"]
        question = row["question"]
        label = 1

        multi_examples = []
        for p in source_page:
            if p not in tables_samepdf.keys():
                logger.warning(
                    "Table detector did not find any table in page {} for file {}".format(
                        p, row["source_file"]
                    )
                )
            else:
                relevant_filename = self.find_relevant_table(answer, p, tables_samepdf)

                example = [company, year, question, answer, relevant_filename, label]

                multi_examples.append(example)

        return multi_examples

    def create_negative_examples(self, row, pos_filename, tables_samepdf):
        """
        Args:
            row (A pandas.Series)
            pos_filename (A str): File name of positive table
            tables_samepdf (A dict):  Key= Page number (A str), value = A list of
                                        table csv paths

        Returns:
            examples (A list of lists or empty list): [[str, int, str, None, str, int],...]
        """
        company = row["company"]
        year = row["year"]
        question = row["question"]
        label = 0

        # If kpi id is valid, this will run
        all_tables_in_samepdf = [k for i in tables_samepdf.values() for k in i]
        negative_files_samepdf = [i for i in all_tables_in_samepdf if i != pos_filename]
        negative_files_diffpdf = [
            i
            for i in list(self.filename_to_stringarr)
            if i != pos_filename and i not in negative_files_samepdf
        ]

        # Getting negative examples from other pdfs if there is insufficient within
        # same pdf
        if self.neg_pos_ratio > len(negative_files_samepdf):
            neg = negative_files_samepdf
            extra_neg = random.sample(
                negative_files_diffpdf, self.neg_pos_ratio - len(neg)
            )
            neg.extend(extra_neg)
        else:
            neg = random.sample(negative_files_samepdf, self.neg_pos_ratio)

        examples = [[company, year, question, None, nf, label] for nf in neg]

        return examples

    def find_relevant_table(self, answer, source_page, tables_samepdf):
        """
        Args:
            answer (A str)
            source_page (A str): E.g. "21"
            tables_samepdf (A dict):  Key= Page number (A str), value = A list of
                                        table csv paths

        Returns:
            (A str): table file name which has highest matching score
                     to the answer.

        """
        related_tables = tables_samepdf[source_page]
        scores = []
        for rel_tab in related_tables:
            score = fuzz.token_set_ratio(
                " ".join(map(str, self.filename_to_stringarr[rel_tab])), answer
            )
            scores.append(score)

        return related_tables[np.argmax(scores)]

    def __obtain_filename_to_strarr(self):
        """
        Returns:
            filename_to_stringarr(a dictionary): Key = table csv
                path (str), value = a list of strings in the table
        """
        extraction_files = [
            f for f in os.listdir(self.extraction_folder) if f.endswith(".csv")
        ]

        # Map csv filename to a list of strings in the table
        filename_to_stringarr = dict()
        for f in extraction_files:
            data = pd.read_csv(os.path.join(self.extraction_folder, f), index_col=0)
            tab_arr = data.values.flatten()
            words = tab_arr[~pd.isnull(tab_arr)]
            filename_to_stringarr[f] = words.tolist()

        return filename_to_stringarr

    def __clean_annotation_file(self, df, annotation_filepath):
        """Returns a clean dataframe after dropping all NaN rows,
            dropping rows which has NaN values in some of the columns
            (refer below), filter by TABLE and exclude certain companies,
            mapping kpi id to question and remove rows with invalid kpi id,
            cleaning source_page to get a list of str removing rows with invalid
            source_page, finally adding annotator's name.

        Args:
            df (A dataframe)
            annotation_filepath (A str or PosixPath)
        """
        # dropping all nan rows
        df = (
            df[self.columns_to_read]
            .dropna(axis=0, how="all", subset=self.columns_to_read)
            .reset_index(drop=True)
        )

        # Drop rows with NaN for all columns except answer
        df = df.dropna(
            axis=0,
            how="any",
            subset=["company", "source_file", "source_page", "kpi_id", "year"],
        ).reset_index(drop=True)

        # Filter dataframe to rows which are table and exclude certain companies
        boolean = df.data_type == self.data_type

        for exclude in self.company_to_exclude:
            boolean = boolean & (df.company != exclude)
        df = df[boolean]

        # Get pdf filename right (don't need to make it a class method)
        def get_pdf_name_right(f):
            if not f.endswith(".pdf"):
                if f.endswith(",pdf"):
                    filename = f.split(",pdf")[0].strip() + ".pdf"
                else:
                    filename = f.strip() + ".pdf"
            else:
                filename = f.split(".pdf")[0].strip() + ".pdf"

            return filename

        df["source_file"] = df["source_file"].apply(get_pdf_name_right)

        # kpi mapping. No need to make it as class method
        def map_kpi(r):
            try:
                question = KPI_MAPPING[float(r)]
            except (KeyError, ValueError):
                question = None

            return question

        df["question"] = df["kpi_id"].apply(map_kpi)
        invalid_kpi = df[df["question"].isna()]["kpi_id"].unique().tolist()
        if len(invalid_kpi) != 0:
            logger.warning(
                "File {} has invalid kpis: {}".format(
                    os.path.basename(annotation_filepath), invalid_kpi
                )
            )

        # Remove examples with invalid kpi
        df = df.dropna(axis=0, subset=["question"]).reset_index(drop=True)

        # Remove examples where source_page can't be parsed
        def clean_page(sp):
            if sp[0] != "[" or sp[-1] != "]":
                return None
            else:
                # Covers multi pages and fix cases like '02'
                return [str(int(i)) for i in sp[1:-1].split(",")]

        temp = df["source_page"].apply(clean_page)
        invalid_source_page = df["source_page"][temp.isna()].unique().tolist()
        if len(invalid_source_page) != 0:
            logger.warning(
                "File {} has invalid source_page format: {}".format(
                    os.path.basename(annotation_filepath),
                    df["source_page"][temp.isna()].unique(),
                )
            )

        df["source_page"] = temp
        df = df.dropna(axis=0, subset=["source_page"]).reset_index(drop=True)

        # Remove examples with incorrect kpi-data_type pair
        def clean_id(r):
            kpi_id = float(r["kpi_id"])

            if r["data_type"] in KPI_CATEGORY[kpi_id]:
                cat = True
            else:
                cat = False

            return cat

        correct_id_bool = df[["kpi_id", "data_type"]].apply(clean_id, axis=1)
        df = df[correct_id_bool].reset_index(drop=True)
        diff = correct_id_bool.shape[0] - df.shape[0]
        if diff > 0:
            logger.info(
                "Drop {} examples for {} due to incorrect kpi-data_type pair".format(
                    diff, annotation_filepath
                )
            )

        df["annotator"] = os.path.basename(annotation_filepath)
        return df

    def __create_table_meta(self):
        """Returns a dictionary of key = pdf file name, value = a dictionary
            where key = page num, value = a list of table csvs from that page

        Returns:
            meta_dict (A dict of dict)
        """
        extraction_files = [
            f for f in os.listdir(self.extraction_folder) if f.endswith(".csv")
        ]

        meta_list = defaultdict(list)

        for f in extraction_files:
            filename = f.split("_page")[0].strip() + ".pdf"
            meta_list[filename].append(f)

        meta_dict = defaultdict(dict)
        for f in meta_list:
            temp = defaultdict(list)
            for c in meta_list[f]:
                page = c.split("_page")[1].split("_")[0]
                temp[page].append(c)
            meta_dict[f] = temp

        return meta_dict

    def process_single_annotation_file(
        self, annotation_filepath, sheet_name="data_ex_in_xls"
    ):
        """Create examples for a single excel file
        Args:
            annotation_filepath (str): Path to the annotated excel file
            sheet_name (A str): Sheet which contains data
        Returns:
            examples (list of lists): List of positive and negative examples
                                      extracted from excel file
        """
        logger.debug("Processing excel file {}".format(annotation_filepath))

        # Excel file corrupted
        try:
            df = pd.read_excel(annotation_filepath, sheet_name=sheet_name)
        except xlrd.biffh.XLRDError as err:
            logger.warning(
                "Trouble reading excel file {}: {}".format(annotation_filepath, err)
            )
            return [[]]

        # Check if df has all columns
        if any([e not in df.columns for e in self.columns_to_read]):
            logger.warning(
                "Excel file {} has missing columns from {}".format(
                    annotation_filepath, self.columns_to_read
                )
            )
            return [[]]

        # clean dataframe
        df = self.__clean_annotation_file(df, annotation_filepath)

        # table_meta contains {pdf_name:{page: list of table csvs, ...}, ...}
        table_meta = self.__create_table_meta()

        examples = []
        for i, row in df.iterrows():
            tables_samepdf = table_meta[row["source_file"]]

            # skip curation process if pdf does not exist
            if len(tables_samepdf) == 0:
                logger.warning("{} was not extracted.".format(row["source_file"]))
                continue

            positive_example = self.create_pos_examples(row.copy(), tables_samepdf)

            # It will be empty if no table found in the page
            if positive_example == []:
                continue

            pos_filename = positive_example[0][4]
            if self.create_neg_samples:
                negative_examples = self.create_negative_examples(
                    row.copy(), pos_filename, tables_samepdf
                )
                examples.extend(negative_examples + positive_example)
            else:
                examples.extend(positive_example)

        return examples
