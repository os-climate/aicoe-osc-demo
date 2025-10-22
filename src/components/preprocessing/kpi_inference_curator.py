"""TextKPIInferenceCurator class def."""

import os
import re
import ast
import json
import logging
from glob import glob as gg

import numpy as np
import pandas as pd

from datetime import date
from fuzzywuzzy import fuzz

from .base_kpi_inference_curator import BaseKPIInferenceCurator
from src.components.utils.qa_utils import aggregate_excels, clean_annotation
from src.components.utils.kpi_mapping import get_kpi_mapping_category

logger = logging.getLogger(__name__)
COL_ORDER = [
    "company",
    "source_file",
    "source_page",
    "kpi_id",
    "year",
    "answer",
    "data_type",
    "relevant_paragraphs",
    "annotator",
    "sector",
]


class TextKPIInferenceCurator(BaseKPIInferenceCurator):
    """Implement KPI inference data curation from abstract class."""

    def __init__(
        self,
        annotation_folder,
        agg_annotation,
        extracted_text_json_folder,
        output_squad_folder,
        kpi_df,
        columns_to_read,
        relevant_text_path=None,
        name="TextKPIInferenceCurator",
    ):
        """
        Initialize with annotations, kpi, input output folders.

        Args:
            annotation_folder (A path): Path to all annotated excels
            agg_annotation (A path): Path to aggregated excel (will create from
                                        annotation_folder if not exist)
            extracted_text_json_folder (A path): Path to all json files
                                                 which contain text extracted
                                                 from pdf
            output_squad_folder (A path): Path to output squad format like data
            relevant_text_path (A path): Path to the the output of text relevant
                                            detector model.
            name (A str)
        """
        super().__init__(name)
        self.data_type = "TEXT"
        self.annotation_folder = annotation_folder
        self.agg_annotation = agg_annotation
        self.extracted_text_json_folder = extracted_text_json_folder
        self.output_squad_folder = output_squad_folder
        self.relevant_text_path = relevant_text_path
        self.columns_to_read = columns_to_read
        self.kpi_mapping_category = get_kpi_mapping_category(kpi_df)

    def read_agg(self):
        """
        Read aggregated annotation csv. If doesn't exist, will create it from annotation folder.

        Returns:
            df (a pd dataframe)
        """
        if not os.path.exists(self.agg_annotation):
            logger.info("{} not available, will create it.".format(self.agg_annotation))
            df = aggregate_excels(self.annotation_folder, self.columns_to_read)
            df = clean_annotation(
                df,
                self.agg_annotation,
                kpi_category=self.kpi_mapping_category["KPI_CATEGORY"],
            )[COL_ORDER]
        else:
            # df = pd.read_csv(self.agg_annotation, header=0, index_col=0)[COL_ORDER]
            # input_fd = open(self.agg_annotation, errors = 'ignore')
            # df = pd.read_csv(input_fd, header=0, index_col=0)[COL_ORDER]
            # input_fd.close()
            df = pd.read_excel(self.agg_annotation, header=0, index_col=0)[COL_ORDER]
            df.loc[:, "source_page"] = df["source_page"].apply(ast.literal_eval)

        return df

    def clean(self, df):
        """
        Return a clean pandas dataframe.

        Args:
            df (A pandas dataframe)
        """

        # map kpi to question
        def map_kpi(r):
            try:
                question = self.kpi_mapping_category["KPI_MAPPING"][float(r["kpi_id"])]
            except (KeyError, ValueError) as e:
                logger.info(e)
                question = None

            if question is not None:
                try:
                    year = int(float(r["year"]))
                except ValueError:
                    year = r["year"]

                if float(r["kpi_id"]) in self.kpi_mapping_category["ADD_YEAR"]:
                    front = question.split("?")[0]
                    question = front + " in year {}?".format(year)

            return question

        df["question"] = df[["kpi_id", "year"]].apply(map_kpi, axis=1)
        df = df.dropna(axis=0, subset=["question"]).reset_index(drop=True)

        # Remove NaN rows based on relevant paragraphs and answer
        df = df[~df["relevant_paragraphs"].isna()]
        df = df[~df["answer"].isna()]

        # change line space to white space, remove trailing and initial white space
        df.loc[:, "answer"] = df["answer"].apply(lambda x: " ".join(str(x).split("\n")))
        df.loc[:, "answer"] = df["answer"].apply(lambda x: x.strip())

        # clean relevant_paragraphs
        df.loc[:, "relevant_paragraphs"] = df["relevant_paragraphs"].apply(
            self.clean_paragraph
        )
        df = df.dropna(axis=0, subset=["relevant_paragraphs"]).reset_index(drop=True)

        # split multiple paragraphs to individual examples
        df = self.split_multi_paragraph(df)

        return df

    def split_multi_paragraph(self, df):
        """Split multiple relevant paragraphs to individual examples."""
        # if single relevant paragraphs, then assuming only has single source page (fair enough)
        df_single = df[df["relevant_paragraphs"].apply(len) == 1].copy()
        df_single.loc[:, "source_page"] = df_single["source_page"].apply(lambda x: x[0])
        df_single.loc[:, "relevant_paragraphs"] = df_single[
            "relevant_paragraphs"
        ].apply(lambda x: x[0])

        # Otherwise
        df_multi = df[df["relevant_paragraphs"].apply(len) > 1]
        new_multi = []

        # better to check before using itertuples
        col_order = COL_ORDER + ["question"]
        assert all(
            [e in df_multi.columns.tolist() for e in COL_ORDER]
        ), "dataframe columns are different. Your df column {}".format(
            df_multi.columns.tolist()
        )

        df_multi = df_multi[col_order]
        for row in df_multi.itertuples():
            # if single source page and multiple relevant paragraphs
            if len(row[3]) == 1:
                for i in range(len(row[8])):
                    new_row = [i for i in row[1:]]
                    new_row[2] = row[3][0]
                    new_row[7] = row[8][i]
                    new_multi.append(new_row)

            # if multiple source pages and multiple relevant paragraphs
            # just do index-to-index matching
            if len(row[3]) > 1:
                for i in range(len(row[3])):
                    new_row = [i for i in row[1:]]
                    new_row[2] = row[3][i]
                    new_row[7] = row[8][i]
                    new_multi.append(new_row)

        df_multi = pd.DataFrame(new_multi, columns=df_multi.columns)
        df = pd.concat([df_single, df_multi], axis=0).reset_index(drop=True)

        return df

    def clean_paragraph(self, r):
        """Clean relevant_paragraphs column.

        Args:
            r (A pandas series row)
        """
        # remove any starting or trailing white spaces
        strp = r.strip()

        # try my best to fix some issues with brackets, parenthesis typos
        if strp[0] == "{" or strp[0] == "]":
            strp = "".join(["["] + list(strp)[1:])
        elif strp[-1] == "}" or strp[-1] == "[":
            strp = "".join(list(strp)[:-1] + ["]"])

        s = strp[0]
        e = strp[-1]

        if s != "[" or e != "]":
            return None  # remove if not able to fix
        else:
            # deal with multiple paragraphs
            # first possible type of split
            strp = strp[2:-2]
            first_type = re.finditer('", "', strp)
            first_type = [i for i in first_type]

            # second
            second_type = re.finditer('","', strp)
            second_type = [i for i in second_type]

            if len(first_type) == 0 and len(second_type) == 0:
                return [strp]
            elif len(first_type) != 0 and len(second_type) == 0:
                temp = []
                start = 0
                for i in first_type:
                    temp.append(strp[start : i.start()])
                    start = i.start() + 4
                temp.append(strp[start:])
                return temp
            elif len(first_type) == 0 and len(second_type) != 0:
                temp = []
                start = 0
                for i in second_type:
                    temp.append(strp[start : i.start()])
                    start = i.start() + 3
                temp.append(strp[start:])
                return temp
            else:  # a combination of two
                temp = []
                start = 0
                track1 = 0
                track2 = 0

                while track1 < len(first_type) or track2 < len(second_type):
                    if track1 == len(first_type):
                        for i in second_type[track2:]:
                            temp.append(strp[start : i.start()])
                            start = i.start() + 3
                            break

                    if track2 == len(second_type):
                        for i in first_type[track1:]:
                            temp.append(strp[start : i.start()])
                            start = i.start() + 4
                            break

                    if first_type[track1].start() < second_type[track2].start():
                        temp.append(strp[start : first_type[track1].start()])
                        start = first_type[track1].start() + 4
                        track1 += 1
                    else:
                        temp.append(strp[start : second_type[track2].start()])
                        start = second_type[track2].start() + 3
                        track2 += 1

                return temp

    def find_closest_paragraph(self, pars, clean_rel_par, clean_answer):
        """
        Find paragraph closest to annotated relevant paragraph.

        Args:
            pars (A list of str)
            clean_rel_par (A str): Annotated clean relevant paragraph
        Return:
            clean_rel_par (A str): Annotated/closest full paragraph
        """
        clean_pars = [self.clean_text(p) for p in pars]
        found = False
        for p in clean_pars:
            sentence_start = self.find_answer_start(clean_rel_par, p)
            if len(sentence_start) != 0:
                clean_rel_par = p
                found = True
                break

        # if can't find exact match (could be due to annotation spelling error)
        if not found:
            scores = [fuzz.partial_ratio(p, clean_rel_par) for p in clean_pars]
            max_par = clean_pars[np.argmax(scores)]
            ans_start = self.find_answer_start(clean_answer, max_par)

            if len(ans_start) != 0:
                # Set relative paragraph to closest full paragraph
                clean_rel_par = max_par

        return clean_rel_par

    def return_full_paragraph(self, r, json_dict):
        """Find closest full paragraph, if can't be found return annotated paragraph instead.

        Args:
            r (A pandas series row)
            json_dict (A dict): {pdf_name: {page:list of paragraph}}
        Returns:
            clean_rel_par (A str)
            clean_answer (A str)
            ans_start (A list of int)
        """
        clean_answer = self.clean_text(r["answer"])
        clean_rel_par = self.clean_text(r["relevant_paragraphs"])

        # If json file not extracted, use relevant text from annotation
        if r["source_file"] not in json_dict:
            logger.info(
                "{} json file has not been extracted. Will use relevant text as annotated.".format(
                    r["source_file"]
                )
            )
        else:
            d = json_dict[r["source_file"]]

            # pdfminer starts counter from 0 (hence the dictionary loaded from json)
            page_num = str(int(r["source_page"]) - 1)
            if page_num not in d:
                logger.info(
                    "{}.json does not have an entry for page {}. \
                    Will use relevant text as annotated".format(
                        r["source_file"].split(".pdf")[0], r["source_page"]
                    )
                )
            else:
                pars = d[page_num]
                if len(pars) == 0:
                    logger.info(
                        "{}.json has empty list of paragraphs at page {}. \
                        Will use relevant text as annotated".format(
                            r["source_file"].split(".pdf")[0], r["source_page"]
                        )
                    )
                else:
                    # match the closest paragraph to the annotated one
                    # let's try exact match
                    clean_rel_par = self.find_closest_paragraph(
                        pars, clean_rel_par, clean_answer
                    )

        ans_start = self.find_answer_start(clean_answer, clean_rel_par)

        # avoid 0th index answer due to FARM bug
        if 0 in ans_start:
            clean_rel_par = " " + clean_rel_par
            ans_start = [i + 1 for i in ans_start]

        return clean_rel_par, clean_answer, ans_start

    def curate(
        self, val_ratio, seed, find_new_answerable=True, create_unanswerable=True
    ):
        """
        Curate squad samples.

        Args:
            val_ratio (A float)
            seed (An int)
            find_new_answerable (A bool): Whether to search for additional
                                            answerable samples
            create_unanswerable (A bool): Whether to create unanswerable
                                            samples
        """
        df = self.read_agg()
        df = df[df["data_type"] == self.data_type]
        df = self.clean(df)

        # get all available jsons from extraction phase
        all_json = [
            i
            for i in os.listdir(self.extracted_text_json_folder)
            if i.endswith(".json")
        ]

        json_dict = {}
        for f in all_json:
            name = f.split(".json")[0]

            with open(os.path.join(self.extracted_text_json_folder, f), "r") as fi:
                d = json.load(fi)
            json_dict[name + ".pdf"] = d

        answerable_df = self.create_answerable(df, json_dict, find_new_answerable)

        if create_unanswerable:
            unanswerable_df = self.create_unanswerable(df)
            all_df = (
                pd.concat([answerable_df, unanswerable_df])
                .drop_duplicates(subset=["answer", "paragraph", "question"])
                .reset_index(drop=True)
            )
        else:
            all_df = answerable_df

        squad_json = self.create_squad_from_df(all_df)
        train_squad, val_squad = self.split_squad(squad_json, val_ratio, seed)

        da = date.today().strftime("%d-%m-%Y")
        # save data as csv for reference
        all_df.to_csv(
            os.path.join(self.output_squad_folder, "reference_kpi_{}.csv".format(da))
        )
        if train_squad != {}:
            train_f = os.path.join(self.output_squad_folder, "kpi_train.json")
            with open(train_f, "w") as f:
                json.dump(train_squad, f)

        if val_squad != {}:
            val_f = os.path.join(self.output_squad_folder, "kpi_val_split.json")
            with open(val_f, "w") as f:
                json.dump(val_squad, f)

        return train_squad, val_squad

    def create_answerable(self, df, json_dict, find_new_answerable):
        """
        Create answerable samples.

        Args:
            df (A dataframe)
            json_dict (A dict): {pdf_name: {page:list of paragraph}}
            find_new_answerable (A boolean)
        Returns:
            pos_df (A dataframe)
        """
        # find closest full paragraph
        results = df.apply(self.return_full_paragraph, axis=1, json_dict=json_dict)

        # set new answer, relevant_paragraphs and add answer_start
        temp = pd.DataFrame(results.tolist())
        df["relevant_paragraphs"] = temp[0]
        df["answer"] = temp[1]
        df["answer_start"] = temp[2]
        df = df[~df["answer"].isna()]

        if find_new_answerable:
            synthetic_pos = self.find_extra_answerable(df, json_dict)
        else:
            synthetic_pos = pd.DataFrame([])

        pos_df = (
            pd.concat([df, synthetic_pos])
            .drop_duplicates(subset=["answer", "relevant_paragraphs", "question"])
            .reset_index(drop=True)
        )

        pos_df = pos_df[pos_df["answer_start"].apply(len) != 0].reset_index(drop=True)
        pos_df.rename({"relevant_paragraphs": "paragraph"}, axis=1, inplace=True)

        pos_df = pos_df[
            ["source_file", "paragraph", "question", "answer", "answer_start"]
        ]

        return pos_df

    def find_extra_answerable(self, df, json_dict):
        """
        Find extra answerable samples.

        Args:
            df (A dataframe)
            json_dict (A dict): {pdf_name: {page:list of paragraph}}
        Returns:
            new_positive_df (A dataframe)
        """
        new_positive = []
        for t in df.itertuples():
            pdf_name = t[2]
            page = str(int(t[3]) - 1)
            clean_answer = self.clean_text(t[6])
            kpi_id = t[4]

            if pdf_name not in json_dict.keys():
                continue

            # skip years questions, company
            if float(kpi_id) in [0, 1, 9, 11]:
                continue

            for p in json_dict[pdf_name].keys():
                if p == page:
                    continue

                pars = json_dict[pdf_name][p]

                if len(pars) == 0:
                    continue

                for par in pars:
                    clean_rel_par = self.clean_text(par)
                    ans_start = self.find_answer_start(clean_answer, clean_rel_par)

                    # avoid 0th index answer due to FARM bug
                    if 0 in ans_start:
                        clean_rel_par = " " + clean_rel_par
                        ans_start = [i + 1 for i in ans_start]

                    if len(ans_start) != 0:
                        example = [
                            t[1],
                            t[2],
                            p,
                            kpi_id,
                            t[5],
                            clean_answer,
                            t[7],
                            clean_rel_par,
                            "1QBit",
                            t[10],
                            t[11],
                            ans_start,
                        ]
                        new_positive.append(example)

            new_positive_df = pd.DataFrame(new_positive, columns=df.columns)

            return new_positive_df

    def create_unanswerable(self, annotation_df):
        """Create unanswerable examples.

        Unanswerable examples are generated from the pair of KPI question and
        pdf's paragraphs that are classified as relevant by the relevance
        detector model, while they are not present in the annotation files.

        Args:
            annotation_df (Pandas.Dataframe): An aggregated dataframe
                                            containing all the annotations.
        Return:
            negative_dataset (dict): A dictionary containing the negative examples,
                                    same format as squad dataset.
        """
        # TODO: creating the logging.

        # read all relevance results into one csv
        relevance_results = gg(str(self.relevant_text_path))

        # Get the relevant pairs of Kpi  questions and paragraphs
        df_list = []
        for fpath in relevance_results:
            df = pd.read_csv(fpath, header=0, index_col=0, usecols=[0, 1, 2, 3, 4])
            df_list.append(df)
        relevant_df = pd.concat(df_list, axis=0, ignore_index=True)

        order_col = ["page", "pdf_name", "text", "text_b"]
        assert all([e in relevant_df.columns for e in order_col])
        relevant_df = relevant_df[order_col]

        def add_pdf_extension(pdf_name):
            # pdf_name = " ".join(pdf_name.split("-")[:-2])
            return str(pdf_name) + ".pdf"

        relevant_df.loc[:, "text_b"] = relevant_df["text_b"].apply(self.clean_text)

        relevant_df.loc[:, "pdf_name"] = relevant_df.apply(
            lambda x: add_pdf_extension(x.pdf_name), axis=1
        )

        # Pages in the json files start from 0, while in a pdf viewer it starts from 1.
        relevant_df.loc[:, "page_viewer"] = relevant_df.apply(
            lambda x: x.page + 1, axis=1
        )

        neg_df = self.filter_relevant_examples(annotation_df, relevant_df)
        neg_df.rename(
            {"text": "question", "text_b": "paragraph", "pdf_name": "source_file"},
            inplace=True,
            axis=1,
        )
        neg_df["answer_start"] = [[]] * neg_df.shape[0]
        neg_df["answer"] = ""
        neg_df = neg_df.drop_duplicates(
            subset=["answer", "paragraph", "question"]
        ).reset_index(drop=True)

        neg_df = neg_df[
            ["source_file", "paragraph", "question", "answer", "answer_start"]
        ]

        return neg_df

    def filter_relevant_examples(self, annotation_df, relevant_df):
        """Filter relevant examples that are mentioned in the annotation files.

        For each source pdf, exclude relative examples which their pages are mentioned in the annotation file.

        Args:
            annotation_df (Pandas.DataFrame): All annotation merged in a single dataframe
            relevant_df (Pandas.DataFrame): Dataframe of relevant examples
        Return:
            merged_neg_examples_df (Pandas.DataFrame): Subset of relevant_df that is
                            considered as negative examples.
        """
        # Get the list of pdfs mention in relevant data frame
        target_pdfs = list(relevant_df["pdf_name"].unique())

        neg_examples_df_list = []
        for pdf_file in target_pdfs:
            annotation_for_pdf = annotation_df[annotation_df["source_file"] == pdf_file]
            if len(annotation_for_pdf) == 0:
                continue

            pages = list(annotation_for_pdf["source_page"].unique())

            neg_examples_df = relevant_df[
                (relevant_df["pdf_name"] == pdf_file)
                & ~(relevant_df["page_viewer"].isin(pages))
            ]

            questions = annotation_for_pdf["question"].tolist()
            answers = annotation_for_pdf["answer"].astype(str).tolist()

            # This is an extra step to make sure the negative examples do not
            # contain the answer of a question.
            for q, a in zip(questions, answers):
                neg_examples_df = neg_examples_df[
                    ~(
                        (neg_examples_df["text"] == q)
                        & (
                            neg_examples_df["text_b"].map(
                                lambda x: self.clean_text(a) in x
                            )
                        )
                    )
                ]

                neg_examples_df_list.append(neg_examples_df)

        merged_neg_examples_df = pd.concat(neg_examples_df_list)

        return merged_neg_examples_df
