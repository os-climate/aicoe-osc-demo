"""Text KPI Inference."""


import glob
import logging
import os
from collections import defaultdict

import pandas as pd
from farm.data_handler.utils import write_squad_predictions
from farm.infer import QAInferencer

from src.components.utils.kpi_mapping import get_kpi_mapping_category

_logger = logging.getLogger(__name__)


def aggregate_result(x):
    """Aggregate result method (helper function).

    Helper function Used in `infer_on_relevance_results`
    For relevant paragraphs related to a single pdf and question, find groups that the answer with highest score
    is always no_answer. If that happens, we consider that question is not answerable for the given pdf.
    """
    rank_1 = x[x["rank"] == "rank_1"]
    aggregated_no_answer = all(rank_1["answer"] == "no_answer")
    if aggregated_no_answer:
        max_no_answer_score = rank_1["score"].max()
        return max_no_answer_score


class TextKPIInfer:
    """TextKPIInfer class.

    This class is responsible for making inference based on the qa system, the qa system is a bert based model
    trained on text data.
    The qa model will be loaded and one of the methods has to be called depending on the input type.
        1. infer_on_dict: The input is a list of dictionaries of questions and context.
        2. infer_on_file: The input is the path to the squad-like file.
        3. infer_on_relevance_results: The input is the path to the predictions of relevance detector model (csv file)
    Args:
        infer_config: (obj of src.config.QAInferConfig)
        n_best_per_sample (int): num candidate answer spans to consider from each passage. Each passage also
            returns "no answer" info. This is the parameter for farm qa model.
    """

    def __init__(self, infer_config, n_best_per_sample=1):
        """Initialize TextKPIInfer class."""
        self.infer_config = infer_config

        farm_logger = logging.getLogger("farm")
        farm_logger.setLevel(self.infer_config.farm_infer_logging_level)

        self.model = QAInferencer.load(
            self.infer_config.load_dir["Text"],
            batch_size=self.infer_config.batch_size,
            gpu=self.infer_config.gpu,
            num_processes=self.infer_config.num_processes,
        )
        # num span-based candidate answer spans to consider from each passage
        self.model.model.prediction_heads[0].n_best_per_sample = n_best_per_sample
        # If positive, this will boost "No Answer" as prediction.
        # If negative, this will decrease the model from giving "No Answer" as prediction.
        self.model.model.prediction_heads[
            0
        ].no_ans_boost = self.infer_config.no_ans_boost
        self.result_dir = self.infer_config.result_dir["Text"]
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def infer_on_dict(self, input_dict):
        """Make inference using the qa model on the input_dictionary.

        Args:
            input_dict: (list of dict)
                example: input_dict = [{"qas": [q1], "context": c1}]
        Returns:
            result (dict): Result dictionary with 'predictions' and 'task' keys.

        """
        result = self.model.inference_from_dicts(dicts=input_dict)
        self.model.close_multiprocessing_pool()
        return result

    def infer_on_file(self, squad_format_file, out_filename="predictions_of_file.json"):
        """Make inference using the qa model on the squad formatted json file.

        Args:
            squad_format_file (str): Path to the file.
            out_filename (str): The file name for the output.
        Returns:
            results (list of farm.modeling.predictions.QAPred): Predictions of the model.
        Note: The result also will be saved in `self.result_dir` directory.
        """
        results = self.model.inference_from_file(
            file=squad_format_file, return_json=False
        )
        result_squad = [x.to_squad_eval() for x in results]

        write_squad_predictions(
            predictions=result_squad,
            predictions_filename=squad_format_file,
            out_filename=os.path.join(self.result_dir, out_filename),
        )
        self.model.close_multiprocessing_pool()
        return results

    def infer_on_relevance_results(self, relevance_results_dir, kpi_df):
        """Make inference using the qa model on the relevant paragraphs.

        Args:
            relevance_results_dir (str): path to the directory where the csv file containing the relevant paragraphs
            and KPIs for text are stored (output from the relevance stage).
            kpi_df (Pandas.DataFrame): A dataframe with kpi questions
        Returns:
            span_df (Pandas.DataFrame): A dataframe, containing best n answers for each KPI question for each pdf.
                The n is defined by top_k. The following columns are added:
                    `answer_span`: answer span
                    `score`: The score of span from qa model
                    `rank`: for the given context and question, what is the rank of score for answer_span. For examples,
                        if rank of a span is rank_1, it means that for the give context and question,
                        the qa model gives the highest score to that span. while rank_2 means, the best guess of model
                        is either`no_answer` or another span.

        Note: The  result data frame will be saved in the `self.result_dir` directory.
        """
        all_relevance_results_paths = glob.glob(
            os.path.join(relevance_results_dir, "*.csv")
        )
        all_span_dfs = []
        num_csvs = len(all_relevance_results_paths)
        _logger.info(
            "{} Starting KPI Inference for the following relevance CSV files found in {}:\n{} ".format(
                "#" * 20,
                self.result_dir,
                [
                    os.path.basename(relevance_results_path)
                    for relevance_results_path in all_relevance_results_paths
                ],
            )
        )
        for i, relevance_results_path in enumerate(all_relevance_results_paths):
            _logger.info("{} {}/{}".format("#" * 20, i + 1, num_csvs))
            pdf_name = os.path.basename(relevance_results_path).split(
                "_predictions_relevant"
            )[0]
            predictions_file_name = "{}_{}".format(pdf_name, "predictions_kpi.csv")
            if (
                self.infer_config.skip_processed_files
                and predictions_file_name in os.listdir(self.result_dir)
            ):
                _logger.info(
                    "The KPI infer results for {} already exists. Skipping.".format(
                        pdf_name
                    )
                )
                _logger.info(
                    "If you would like to re-process the already processed files, set "
                    "`skip_processed_files` to False in the config file. "
                )
                continue
            _logger.info("Starting KPI Extraction for {}".format(pdf_name))
            input_df = pd.read_csv(relevance_results_path)
            column_names = ["text_b", "text", "page", "pdf_name", "source"]

            if len(input_df) == 0:
                _logger.info(
                    "The received relevance file is empty for {}".format(pdf_name)
                )
                df_empty = pd.DataFrame([])
                df_empty.to_csv(os.path.join(self.result_dir, predictions_file_name))
                continue

            assert set(column_names).issubset(
                set(input_df.columns)
            ), """The result of relevance detector has {} columns,
            while expected {}""".format(
                input_df.columns, column_names
            )

            qa_dict = [
                {"qas": [question], "context": context}
                for question, context in zip(input_df["text"], input_df["text_b"])
            ]
            num_data_points = len(qa_dict)
            result = []
            chunk_size = 1000
            chunk_idx = 0
            while chunk_idx * chunk_size < num_data_points:
                data_chunk = qa_dict[
                    chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size
                ]
                predictions_chunk = self.model.inference_from_dicts(dicts=data_chunk)
                result.extend(predictions_chunk)
                chunk_idx += 1
            # result = self.model.inference_from_dicts(dicts=qa_dict)

            head_num = 0
            num_answers = self.model.model.prediction_heads[0].n_best_per_sample + 1
            answers_dict = defaultdict(list)

            for exp in result:
                preds = exp["predictions"][head_num]["answers"]
                # Get the no_answer_score
                no_answer_score = [
                    p["score"] for p in preds if p["answer"] == "no_answer"
                ]
                if (
                    len(no_answer_score) == 0
                ):  # Happens if no answer is not among the n_best predictions.
                    no_answer_score = (
                        preds[0]["score"] - exp["predictions"][head_num]["no_ans_gap"]
                    )
                else:
                    no_answer_score = no_answer_score[0]

                # Based on Farm implementation, no_answer_score already is equal = "CLS score" + no_ans_boost
                # https://github.com/deepset-ai/FARM/blob/978da5d7600c48be458688996538770e9334e71b/farm/modeling/prediction_head.py#L1348
                pure_no_ans_score = no_answer_score - self.infer_config.no_ans_boost

                for i in range(
                    num_answers
                ):  # This param is not exactly representative, n_best mostly defines num answers.
                    answers_dict[f"rank_{i+1}"].append(
                        (
                            preds[i]["answer"],
                            preds[i]["score"],
                            pure_no_ans_score,
                            no_answer_score,
                        )
                    )
            for i in range(num_answers):
                input_df[f"rank_{i+1}"] = answers_dict[f"rank_{i+1}"]

            # Let's put different kpi predictions and their scores into one column so we can sort them.
            var_cols = [i for i in list(input_df.columns) if i.startswith("rank_")]
            id_vars = [i for i in list(input_df.columns) if not i.startswith("rank_")]
            input_df = pd.melt(
                input_df,
                id_vars=id_vars,
                value_vars=var_cols,
                var_name="rank",
                value_name="answer_score",
            )

            # Separate a column with tuple value into two columns
            input_df[
                ["answer", "score", "no_ans_score", "no_answer_score_plus_boost"]
            ] = pd.DataFrame(input_df["answer_score"].tolist(), index=input_df.index)
            input_df = input_df.drop(columns=["answer_score"], axis=1)

            no_answerables = (
                input_df.groupby(["pdf_name", "text"])
                .apply(lambda grp: aggregate_result(grp))
                .dropna(how="all")
            )
            no_answerables = pd.DataFrame(
                no_answerables, columns=["score"]
            ).reset_index()
            no_answerables["answer"] = "no_answer"
            no_answerables["source"] = "Text"

            # Filter to span-based answers
            span_df = input_df[input_df["answer"] != "no_answer"]
            # Concatenate the result of span answers with non answerable examples.
            span_df = pd.concat([span_df, no_answerables], ignore_index=True)

            # Get the predictions with n highest score for each pdf and question.
            # If the question is considered unanswerable, the best prediction is "no_answer", but the best span-based answer
            # is also returned. if the question is answerable, the best span-based answers are returned.
            span_df = (
                span_df.groupby(["pdf_name", "text"])
                .apply(lambda grp: grp.nlargest(self.infer_config.top_k, "score"))
                .reset_index(drop=True)
            )

            # Final cleaning on the dataframe, removing unnecessary columns and renaming `text` and `text_b` columns.
            unnecessary_cols = ["rank"] + [
                i for i in list(span_df.columns) if i.startswith("Unnamed")
            ]
            span_df = span_df.drop(columns=unnecessary_cols, axis=1)
            span_df.rename(columns={"text": "kpi", "text_b": "paragraph"}, inplace=True)

            # Add the kpi id
            reversed_kpi_mapping = {
                value[0]: key
                for key, value in get_kpi_mapping_category(kpi_df)[
                    "KPI_MAPPING"
                ].items()
            }
            span_df["kpi_id"] = span_df["kpi"].map(reversed_kpi_mapping)

            # Change the order of columns
            first_cols = ["pdf_name", "kpi", "kpi_id", "answer", "page"]
            column_order = first_cols + [
                col for col in span_df.columns if col not in first_cols
            ]
            span_df = span_df[column_order]

            result_path = os.path.join(self.result_dir, predictions_file_name)
            span_df.to_csv(result_path)
            _logger.info("Save the result of KPI extraction to {}".format(result_path))
            all_span_dfs.append(span_df)
        concatenated_dfs = (
            pd.concat(all_span_dfs) if len(all_span_dfs) > 0 else pd.DataFrame()
        )
        self.model.close_multiprocessing_pool()
        return concatenated_dfs
