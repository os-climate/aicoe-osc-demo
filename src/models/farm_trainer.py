"""FARM Trainer."""

import logging
import os

import numpy as np
import pandas as pd
from farm.data_handler.data_silo import DataSilo, DataSiloForCrossVal
from farm.data_handler.processor import Processor
from farm.eval import Evaluator
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from .trainer_optuna import TrainerOptuna

_logger = logging.getLogger(__name__)


class FARMTrainer:
    """FARMTrainer class."""

    def __init__(
        self,
        file_config,
        tokenizer_config,
        processor_config,
        training_config,
        mlflow_config,
        model_config,
    ):
        """Initialize FARM Trainer class.

        This class initialize framework components and build a pipeline to train
        or fine-tune a transformers based model
        :param file_config: config object which sets paths
        :param tokenizer_config: config object which sets FARM tokenizer parameters
        :param processor_config: config object which sets FARM processor parameters
        :param training_config: config object which sets training parameters
        :param mlflow_config: config object which sets MLflow parameters to monitor training
        :param model_config: config object which sets FARM AdaptiveModel parameters
        """
        super().__init__()
        self.file_config = file_config
        self.tokenizer_config = tokenizer_config
        self.processor_config = processor_config
        self.model_config = model_config
        self.training_config = training_config
        self.mlflow_config = mlflow_config

    def prepare_data(self):
        """Split data between training set and development set.

        Data are split according to split ratio and save sets to .csv files
        """
        if os.path.exists(self.file_config.train_filename) and os.path.exists(
            self.file_config.dev_filename
        ):
            pass

        data = pd.read_csv(self.file_config.curated_data)
        data = data[["question", "context", "label"]]
        data["text"] = data["question"]
        data["text_b"] = data["context"]
        data.drop(columns=["question", "context"], inplace=True)
        data.dropna(how="any", inplace=True)
        data.drop_duplicates(inplace=True)
        data = shuffle(data)
        data_train, data_dev = train_test_split(
            data, test_size=self.file_config.dev_split
        )

        data_train.to_csv(self.file_config.train_filename)
        data_dev.to_csv(self.file_config.dev_filename)

    def _gather_text_from_table(self, file):
        table = pd.read_csv(file, index_col=0)
        text = []
        columns = list(table)
        for column in columns:
            column_proces = table[column].dropna().astype(str)
            numbers = column_proces.str.findall(r"^\W*[0-9]*\W?[0-9]*?\W*$")
            not_numbers = numbers.apply(lambda x: False if len(x) > 0 else True)
            not_numbers_idx = not_numbers[not_numbers == True].index  # noqa: E712
            column_text = [text for text in column_proces[not_numbers_idx]]
            text += column_text
        return ", ".join(text)

    def _prepare_table_data(self):
        if os.path.exists(self.file_config.train_filename) and os.path.exists(
            self.file_config.dev_filename
        ):
            pass

        curated = pd.read_csv(self.file_config.curated_table_data)
        curated_files = [
            os.path.join(self.file_config.extracted_tables_dir, file)
            for file in curated["Table_filename"]
        ]
        data_text = list(map(self._gather_text_from_table, curated_files))
        curated["text_b"] = data_text
        curated["text"] = curated["Question"]
        curated["label"] = curated["Label"]

        data = curated[["text", "text_b", "label"]]
        data.dropna(how="any", inplace=True)
        data.drop_duplicates(inplace=True)

        # drop length 0 tables
        data_len = data["text_b"].apply(len)
        data = data[data_len > 0]

        data = shuffle(data)
        data_train, data_dev = train_test_split(
            data, test_size=self.file_config.dev_split
        )

        data_train.to_csv(self.file_config.train_filename)
        data_dev.to_csv(self.file_config.dev_filename)

    def create_tokenizer(self):
        """Load FARM tokenizer according to language model specified in config object.

        :return FARM tokenizer instance
        """
        tokenizer = Tokenizer.load(**self.tokenizer_config.__dict__)
        return tokenizer

    def create_processor(self, tokenizer):
        """Create FARM processor instance.

        Loads the class of processor specified by processor name in corresponding config object.
        Infers the specific type of Processor from a config file placed in the specified directory
        and loads an instance of it.
        :param tokenizer: FARM tokenizer object
        :return: FARM processor instance
        """
        processor = Processor.load(
            tokenizer=tokenizer,
            **{**self.file_config.__dict__, **self.processor_config.__dict__},
        )
        return processor

    def create_silo(self, processor):
        """Create a FARM DataSilo instance.

        It generates and stores PyTorch DataLoader objects for the train, dev and test datasets.
        Relies upon functionality in the processor to do the conversion of the data.
        It will also calculate and display some statistics.
        :param processor: A dataset specific FARN Processor instance which will
        turn input into a Pytorch Dataset.
        :return data_silo: DataSilo instance,
        :return n_batches (int) number of batches for training
        """
        data_silo = DataSilo(
            processor=processor,
            batch_size=self.training_config.batch_size,
            distributed=self.training_config.distributed,
            max_processes=self.file_config.num_processes,
        )
        n_batches = len(data_silo.loaders["train"])
        return data_silo, n_batches

    def create_head(self):
        """Create a FARM Prediction Head.

        It takes word embeddings from a language model and generates logits for a given task.
        Can also convert logits to loss and and logits to predictions.
        """
        prediction_head_cls = self.model_config.class_type
        prediction_head = prediction_head_cls(**self.model_config.head_config)
        return prediction_head

    def create_model(self, prediction_head, n_batches, device):
        """Create FARM AdaptiveModel and initialize an optimizer.

        1. Creates FARM AdaptiveModel instance which combines a language model
          and a prediction head.Allows for gradient flow back to the language model
          component.
        2. Initializes an optimizer, a learning rate scheduler and converts the
        model if needed (e.g for mixed precision). Per default, we use transformers'
        AdamW and a linear warmup schedule with warmup ratio 0.1.

        :param prediction_head:
        :param n_batches (int): number of batches for training
        :param device (torch.device): torch.device instance to run model either
                                      on cpu or gpu
        :return model: FARM AdaptiveModel instance
        :return optimizer: optimizer
        :return lr_schedule: learning rate scheduler
        """
        language_model = LanguageModel.load(self.model_config.lang_model)
        model = AdaptiveModel(
            language_model=language_model,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=self.training_config.dropout,
            lm_output_types=self.model_config.lm_output_types,
            device=device,
        )
        if self.model_config.load_dir:
            model = model.load(self.model_config.load_dir, device=device, strict=False)

        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            learning_rate=self.training_config.learning_rate,
            device=device,
            n_batches=n_batches,
            n_epochs=self.training_config.n_epochs,
            grad_acc_steps=self.training_config.grad_acc_steps,
        )
        return model, optimizer, lr_schedule

    def create_trainer(self, model, optimizer, lr_schedule, data_silo, device, n_gpu):
        """Create FARM Trainer instance that handles the main model training procedure.

        This includes performing evaluation on the dev set at regular
        intervals during training as well as evaluation on the test set at the
        end of training.
        If hyperparameter tuning mode is on, it creates a modified version of
        the FARM Trainer class which allows to stop early trials that are likely
        to be unsuccessful

        :param model: FARM AdaptiveModel instance
        :param optimizer: FARM optimizer instance
        :param lr_schedule: FARM learning rate scheduler instance
        :param data_silo: FARM DataSilo instance
        :param device: torch.device instance to run model either on cpu or gpu
        :param n_gpu: number of gpus available for training and evaluation
        :return trainer: FARM Trainer instance
        """
        if self.training_config.run_hyp_tuning == True:  # noqa: E712
            trainer = TrainerOptuna(
                epochs=self.training_config.n_epochs,
                model=model,
                optimizer=optimizer,
                data_silo=data_silo,
                n_gpu=n_gpu,
                lr_schedule=lr_schedule,
                evaluate_every=self.training_config.evaluate_every,
                device=device,
            )
        else:
            trainer = Trainer(
                epochs=self.training_config.n_epochs,
                model=model,
                optimizer=optimizer,
                data_silo=data_silo,
                n_gpu=n_gpu,
                lr_schedule=lr_schedule,
                evaluate_every=self.training_config.evaluate_every,
                device=device,
                grad_acc_steps=self.training_config.grad_acc_steps,
            )
        return trainer

    def _train_on_split(self, data_silo, silo_to_use, num_fold, device, n_gpu):
        """Train a model for one split of cross-validation.

        :param data_silo: DataSilo instance used for training without CV
        :param silo_to_use: instance of FARM DataSiloForCrossVal the corresponding
        fold
        :param num_fold: (int) fold number
        :param device: torch.device instance to run model either on cpu or gpu
        :param n_gpu: number of gpus available for training and evaluation
        :return: FARM AdaptiveModel instance trained on the corresponding split
        """
        _logger.info(f"############ Crossvalidation: Fold {num_fold} ############")
        prediction_head = self.create_head()
        model, optimizer, lr_schedule = self.create_model(
            prediction_head, n_batches=len(silo_to_use.loaders["train"]), device=device
        )
        model.connect_heads_with_processor(
            data_silo.processor.tasks, require_labels=True
        )
        trainer = self.create_trainer(
            model, optimizer, lr_schedule, silo_to_use, device, n_gpu
        )
        trainer.train()
        return trainer.model

    def post_process_dev_results(self, result):
        """Post-process dev results."""
        return result[0]

    def run_cv(self, data_silo, xval_folds, device, n_gpu):
        """Train the model and evaluates it for each fold of cross-validation.

        For performing cross validation, we really want to combine all the
        instances from all the sets or just some of the sets, then create a
        different data silo instance for each fold.
        Calling DataSiloForCrossVal.make() creates a list of DataSiloForCrossVal
        instances - one for each fold.

        Mean F1-score, mean accuracy, mean recall and mean precision are computed
        for validation

        :param data_silo:
        :param xval_folds: (int) number of folds for cross-validation
        :param device: torch.device instance to run model either on cpu or gpu
        :param n_gpu: number of gpus available for training and evaluation
        :return:
        """
        all_f1 = []
        all_recall = []
        all_accuracy = []
        all_precision = []
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
            preds = [int(pred) for pred in result[0].get("preds")]
            labels = [int(label) for label in result[0].get("labels")]
            all_f1.append(f1_score(preds, labels))
            all_recall.append(recall_score(preds, labels))
            all_accuracy.append(result[0]["acc"])
            all_precision.append(precision_score(preds, labels))
        _logger.info(
            f"############ RESULT_CV -- {self.training_config.xval_folds} folds ############"
        )
        _logger.info(
            f"Mean F1:  {np.mean(all_f1)*100:.1f}, std F1: {np.std(all_f1):.3f}"
        )
        _logger.info(
            f"Mean recall:  {np.mean(all_recall)*100:.1f}, std recall: {np.std(all_recall):.3f}"
        )
        _logger.info(
            f"Mean accuracy:  {np.mean(all_accuracy)*100:.1f}, std accuracy; {np.std(all_accuracy):.3f}"
        )
        _logger.info(
            f"Mean precision:  {np.mean(all_precision)*100:.1f}, std  precision: {np.std(all_precision):.3f}"
        )

    def run(self, trial=None, metric="acc"):
        """Run model training using created FARM component.

        Saves a checkpoint for model and processor if hyperparameter tuning or cross-validation modes
        are not on. If hyperparameter tuning mode is specified in corresponding config object,
        this methods will run a trial defined by Optuna hyperparameter tuning framework
        :param trial: Optuna trial instance for hyperparameter tuning corresponding
        to training for one combination of hyperparameters
        :return Accuracy of training:
        """
        set_all_seeds(self.training_config.seed)
        if self.mlflow_config.track_experiment:
            ml_logger = MLFlowLogger(tracking_uri=self.mlflow_config.url)
            ml_logger.init_experiment(
                experiment_name=self.mlflow_config.experiment_name,
                run_name=self.mlflow_config.run_name,
            )

        if self.file_config.data_type == "Text":
            self.prepare_data()
        elif self.file_config.data_type == "Table":
            self._prepare_table_data()
        else:
            raise ValueError("data_type must be set to either `Text` or `Table`.")

        device, n_gpu = initialize_device_settings(
            use_cuda=self.training_config.use_cuda, use_amp=self.training_config.use_amp
        )
        tokenizer = self.create_tokenizer()
        processor = self.create_processor(tokenizer)
        data_silo, n_batches = self.create_silo(processor)

        if self.training_config.run_hyp_tuning:
            prediction_head = self.create_head()
            model, optimizer, lr_schedule = self.create_model(
                prediction_head, n_batches, device
            )
            trainer = self.create_trainer(
                model, optimizer, lr_schedule, data_silo, device, n_gpu
            )
            trainer.train(trial)
            evaluator_dev = Evaluator(
                data_loader=data_silo.get_data_loader("dev"),
                tasks=data_silo.processor.tasks,
                device=device,
            )
            result = evaluator_dev.eval(model, return_preds_and_labels=True)
            evaluator_dev.log_results(
                result, "DEV", logging=True, steps=len(data_silo.get_data_loader("dev"))
            )

        elif self.training_config.run_cv:
            self.run_cv(data_silo, self.training_config.xval_folds, device, n_gpu)

        else:
            prediction_head = self.create_head()
            model, optimizer, lr_schedule = self.create_model(
                prediction_head, n_batches, device
            )
            trainer = self.create_trainer(
                model, optimizer, lr_schedule, data_silo, device, n_gpu
            )
            trainer.train()

            evaluator_dev = Evaluator(
                data_loader=data_silo.get_data_loader("dev"),
                tasks=data_silo.processor.tasks,
                device=device,
            )
            result = evaluator_dev.eval(model, return_preds_and_labels=True)
            evaluator_dev.log_results(
                result, "DEV", logging=True, steps=len(data_silo.get_data_loader("dev"))
            )

            result = self.post_process_dev_results(result)

            model.save(self.file_config.saved_models_dir)
            processor.save(self.file_config.saved_models_dir)
            _logger.info(f"Trained model saved to {self.file_config.saved_models_dir}")
            _logger.info(
                f"Processor vocabulary saved to {self.file_config.saved_models_dir}"
            )
            return result[metric]
