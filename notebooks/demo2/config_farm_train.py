"""Config FARM Training."""

import pathlib
import os
from farm.modeling.prediction_head import TextClassificationHead
import torch
from logging import getLogger, WARNING, INFO, DEBUG
import yaml

with open("settings.yaml", "r") as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)


_logger = getLogger(__name__)
LOGGING_MAPPING = {"info": INFO, "warning": WARNING, "debug": DEBUG}


class Config:
    """Config class."""

    def __init__(self, project_name, experiment_type="RELEVANCE"):
        """Initialize Config class."""
        self.root = pathlib.Path(".").resolve().parent.parent
        self.experiment_type = experiment_type
        self.experiment_name = project_name  # "test_farm"
        self.data_type = "Text"  # Text | Table
        self.seed = settings["config"]["seed"]
        farm_infer_logging_level = "warning"  # FARM logging level during inference; supports info, warning, debug
        self.farm_infer_logging_level = LOGGING_MAPPING[farm_infer_logging_level]


class FileConfig(Config):
    """FileConfig class."""

    def __init__(self, project_name):
        """Initialize FileConfig class."""
        super().__init__(project_name)
        self.data_dir = os.path.join(self.root, "data")
        self.curated_data = os.path.join(
            self.data_dir, "curation", "esg_TEXT_dataset.csv"
        )
        self.curated_table_data = os.path.join(
            self.data_dir, "curation", "esg_TABLE_dataset.csv"
        )
        self.extracted_tables_dir = os.path.join(self.data_dir, "extraction")
        self.dev_split = settings["train_relevance"]["processor"]["proc_dev_split"]
        self.train_filename = os.path.join(self.data_dir, "kpi_train_split.csv")
        self.dev_filename = os.path.join(self.data_dir, "kpi_val_split.csv")
        self.test_filename = None
        self.saved_models_dir = os.path.join(self.root, "models", "RELEVANCE")
        self.num_processes = settings["train_relevance"]["training"]["max_processes"]


class TokenizerConfig(Config):
    """TokenizerConfig class."""

    def __init__(self, project_name):
        """Initialize TokenizerConfig class."""
        super().__init__(project_name)
        self.pretrained_model_name_or_path = settings["train_relevance"][
            "tokenizer_base_model"
        ]
        self.do_lower_case = False


class ProcessorConfig(Config):
    """ProcessorConfig class."""

    def __init__(self, project_name):
        """Initialize ProcessorConfig class."""
        super().__init__(project_name)
        if self.experiment_type == "RELEVANCE":
            self.processor_name = "TextPairClassificationProcessor"
        else:
            raise ValueError("No existing processor for this task")
        self.load_dir = os.path.join(
            self.root, "saved_models", self.data_type, "relevance_roberta"
        )
        # set to None if you don't want to load the\
        # vocab.json file
        self.max_seq_len = settings["train_relevance"]["processor"]["proc_max_seq_len"]
        self.dev_split = settings["train_relevance"]["processor"]["proc_dev_split"]
        self.label_list = settings["train_relevance"]["processor"]["proc_label_list"]
        self.label_column_name = settings["train_relevance"]["processor"][
            "proc_label_column_name"
        ]  # label column name in data files
        self.delimiter = settings["train_relevance"]["processor"]["proc_delimiter"]
        self.metric = settings["train_relevance"]["processor"]["proc_metric"]


class ModelConfig(Config):
    """ModelConfig class."""

    def __init__(self, project_name):
        """Initialize ModelConfig class."""
        super().__init__(project_name)
        if self.experiment_type == "RELEVANCE":
            self.class_type = TextClassificationHead
            self.head_config = {"num_labels": 2}
        else:
            raise ValueError("No existing model for this task")
        # set to None if you don't want to load the config file for this model
        self.load_dir = os.path.join(
            self.root, "models", "relevance_roberta"
        )  # relevance_roberta | relevance_roberta_table_headers
        self.lang_model = settings["train_relevance"]["model"]["model_lang_model"]
        self.layer_dims = settings["train_relevance"]["model"]["model_layer_dims"]
        self.lm_output_types = settings["train_relevance"]["model"][
            "model_lm_output_types"
        ]  # or ["per_tokens"]


class TrainingConfig(Config):
    """TrainingConfig class."""

    def __init__(self, project_name):
        """Initialize TrainingConfig class."""
        super().__init__(project_name)
        self.run_hyp_tuning = settings["train_relevance"]["training"]["run_hyp_tuning"]
        self.use_cuda = True

        # Check if GPU exists
        if not torch.cuda.is_available():
            _logger.warning("No gpu available, setting use_cuda to False")
            self.use_cuda = False

        self.use_amp = settings["train_relevance"]["training"]["use_amp"]
        self.distributed = settings["train_relevance"]["training"]["distributed"]
        self.learning_rate = settings["train_relevance"]["training"]["learning_rate"]
        self.n_epochs = settings["train_relevance"]["training"]["n_epochs"]
        self.evaluate_every = settings["train_relevance"]["training"]["evaluate_every"]
        self.dropout = settings["train_relevance"]["training"]["dropout"]
        self.batch_size = settings["train_relevance"]["training"]["batch_size"]
        self.grad_acc_steps = settings["train_relevance"]["training"]["grad_acc_steps"]
        self.run_cv = settings["train_relevance"]["training"]["run_cv"]
        self.xval_folds = settings["train_relevance"]["training"]["xval_folds"]


class MLFlowConfig(Config):
    """MLFlowConfig class."""

    def __init__(self, project_name):
        """Initialize MLFlowConfig class."""
        super().__init__(project_name)
        self.track_experiment = False
        self.run_name = self.experiment_name
        self.url = "http://localhost:5000"


class InferConfig(Config):
    """InferConfig class."""

    def __init__(self, project_name):
        """Initialize InferConfig class."""
        super().__init__(project_name)
        # please change the following accordingly
        self.data_types = ["Text"]  # ["Text", "Table"] supported "Text", "Table"
        self.load_dir = {"Text": os.path.join(self.root, "models", "RELEVANCE")}
        self.result_dir = {"Text": os.path.join(self.root, "data", "infer_relevance")}
        self.skip_processed_files = settings["infer_relevance"]["skip_processed_files"]
        self.batch_size = settings["infer_relevance"]["batch_size"]
        self.gpu = settings["infer_relevance"]["gpu"]
        self.num_processes = settings["infer_relevance"]["num_processes"]
        # Set to None to let Inferencer use all CPU cores minus one.
        self.disable_tqdm = settings["infer_relevance"]["disable_tqdm"]
        self.extracted_dir = os.path.join(self.root, "data", "extraction")
        self.kpi_questions = settings["infer_relevance"]["kpi_questions"]
        # set to  ["OG", "CM", "CU"] for KPIs of all sectors.
        self.sectors = settings["infer_relevance"]["sectors"]
