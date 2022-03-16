"""QA Farm trainer."""


import logging
import json
import os
import random
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

from farm.data_handler.data_silo import DataSiloForCrossVal
from farm.eval import Evaluator
from farm.modeling.prediction_head import QuestionAnsweringHead
from src.components.utils.qa_metrics import compute_extra_metrics
from .farm_trainer import FARMTrainer

_logger = logging.getLogger(__name__)


class QAFARMTrainer(FARMTrainer):
    """QAFARMTrainer class."""

    def __init__(
        self,
        file_config,
        tokenizer_config,
        processor_config,
        training_config,
        mlflow_config,
        model_config,
    ):
        """Initialize QAFARMTrainer class.

        This class uses FARM framework components and build a pipeline to train
        or fine-tune a transformers based model for a QA task.
        :param file_config: config object which sets paths
        :param tokenizer_config: config object which sets FARM tokenizer parameters
        :param processor_config: config object which sets FARM processor parameters
        :param training_config: config object which sets training parameters
        :param mlflow_config: config object which sets MLflow parameters to monitor training
        :param model_config: config object which sets FARM AdaptiveModel parameters
        """
        super().__init__(
            file_config=file_config,
            tokenizer_config=tokenizer_config,
            processor_config=processor_config,
            model_config=model_config,
            training_config=training_config,
            mlflow_config=mlflow_config,
        )

        if self.file_config.data_type != "Text":
            raise ValueError("only `Text` is supported for QA.")

    def prepare_data(self):
        """Prepare data."""
        if self.file_config.perform_splitting:
            _logger.info(
                "Loading the {} data and splitting to train and val...".format(
                    self.file_config.curated_data
                )
            )

            with open(self.file_config.curated_data, "r") as read_file:
                dataset = json.load(read_file)
            # Splitting
            # create a list of all paragraphs, The splitting will happen at the paragraph level
            paragraphs_list = [
                {"title": pdf["title"], "single_paragraph": par}
                for pdf in dataset["data"]
                for par in pdf["paragraphs"]
            ]
            random.seed(self.file_config.seed)
            random.shuffle(paragraphs_list)
            train_paragraphs, dev_paragraphs = train_test_split(
                paragraphs_list, test_size=self.file_config.dev_split
            )

            def reformat_paragraphs(paragraphs):
                # Function responsible to put the paragraphs related to a same pdf in a the same list
                paragraphs_dict = defaultdict(list)

                for par in paragraphs:
                    paragraphs_dict[par["title"]].append(par["single_paragraph"])

                squad_like_dataset = [
                    {"title": key, "paragraphs": value}
                    for key, value in paragraphs_dict.items()
                ]

                return squad_like_dataset

            train_data = {
                "version": "v2.0",
                "data": reformat_paragraphs(train_paragraphs),
            }
            dev_data = {"version": "v2.0", "data": reformat_paragraphs(dev_paragraphs)}

            with open(self.file_config.train_filename, "w") as outfile:
                json.dump(train_data, outfile)

            with open(self.file_config.dev_filename, "w") as outfile:
                json.dump(dev_data, outfile)
        else:
            _logger.info(
                "Loading the train from {} \n Loading validation data from {}".format(
                    self.file_config.train_filename, self.file_config.dev_filename
                )
            )
            for filename in [
                self.file_config.train_filename,
                self.file_config.dev_filename,
            ]:
                assert os.path.exists(filename), "File `{}` does not exist.".format(
                    filename
                )

    def create_head(self):
        """Create head."""
        if "squad" in self.model_config.lang_model:
            return QuestionAnsweringHead.load(self.model_config.lang_model)
        else:
            return super().create_head()

    def post_process_dev_results(self, result):
        """Post-process dev results."""
        extended_result = compute_extra_metrics(result)
        _logger.info("Extended Results:")
        _logger.info(extended_result)
        return {**extended_result, **result[0]}

    def run_cv(self, data_silo, xval_folds, device, n_gpu):
        """Train the model and evaluates it for each fold of cross-validation.

        Calling DataSiloForCrossVal.make() creates a list of DataSiloForCrossVal
        instances - one for each fold.

        Mean F1-score, EM (overall and answerables only), relaxed_f1_answerable, and top-n accuracy are computed for
         validation.

        :param data_silo:
        :param xval_folds: (int) number of folds for cross-validation
        :param device: torch.device instance to run model either on cpu or gpu
        :param n_gpu: number of gpus available for training and evaluation
        :return:
        """
        all_f1 = []
        all_em = []
        all_top_n_accuracy = []
        all_relaxed_f1_answerable = []
        all_em_answerable = []
        all_f1_answerable = []

        silos = DataSiloForCrossVal.make(
            data_silo, sets=["train", "dev"], n_splits=xval_folds
        )

        for num_fold, silo in enumerate(silos):
            model = self._train_on_split(data_silo, silo, num_fold, device, n_gpu)

            # do eval on test set here (and not in Trainer),
            # so that we can easily store the actual preds and labels for a "global" eval across all folds.
            evaluator_test = Evaluator(
                data_loader=silo.get_data_loader("test"),
                tasks=silo.processor.tasks,
                device=device,
            )
            result = evaluator_test.eval(model, return_preds_and_labels=True)
            result = self.post_process_dev_results(result)
            all_em.append(result["EM"])
            all_f1.append(result["f1"])
            all_top_n_accuracy.append(result["top_n_accuracy"])
            all_relaxed_f1_answerable.append(result["relaxed_f1_answerable"])
            all_em_answerable.append(result["em_answerable"])
            all_f1_answerable.append(result["f1_answerable"])

        _logger.info(
            f"############ RESULT_CV -- {self.training_config.xval_folds} folds ############"
        )
        _logger.info(
            f"EM\nMean:  {np.mean(all_em) * 100:.1f}, std: {np.std(all_em) * 100:.3f}"
        )
        _logger.info(
            f"F1\nMean:  {np.mean(all_f1) * 100:.1f}, std F1: {np.std(all_f1) * 100:.3f}"
        )
        _logger.info(
            f"EM_Answerable\nMean:  {np.mean(all_em_answerable) * 100:.1f}, std recall: {np.std(all_em_answerable) * 100:.3f}"
        )
        _logger.info(
            f"F1_Answerable\nMean:  {np.mean(all_f1_answerable) * 100:.1f}, std F1: {np.std(all_f1_answerable) * 100:.3f}"
        )
        _logger.info(
            f"Relaxed_F1_Answerable\nMean:  {np.mean(all_relaxed_f1_answerable) * 100:.1f}, std F1: "
            f"{np.std(all_relaxed_f1_answerable) * 100:.3f}"
        )
        _logger.info(
            f"Top N Accuracy\nMean:  {np.mean(all_top_n_accuracy)*100:.1f}, std accuracy; "
            f"{np.std(all_top_n_accuracy) * 100:.3f}"
        )
