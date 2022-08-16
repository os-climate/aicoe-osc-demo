"""Config QA FARM Training."""

import logging
import os

import torch
from farm.modeling.prediction_head import QuestionAnsweringHead

from config_farm_train import Config
import yaml

_logger = logging.getLogger(__name__)
with open("settings.yaml", "r") as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)

# E.g. bert-base-uncased, roberta-base, roberta-large, albert-base-v2, albert-large-v2
# or any hugging face model e.g. deepset/roberta-base-squad2, a-ware/roberta-large-squadv2
# Full list at huggingface.co/models
base_lm_model = settings["train_kpi"]["base_model"]
project_name = ""


class QAConfig(Config):
    """QAConfig class."""

    def __init__(self, project_name):
        """Initialize QAConfig class."""
        super().__init__(experiment_type="KPI_EXTRACTION", project_name=project_name)
        self.experiment_name = project_name
        self.data_type = "Text"


class QAFileConfig(QAConfig):
    """QAFileConfig class."""

    def __init__(self, project_name):
        """Initialize QAFileConfig class."""
        super().__init__(project_name)
        self.data_dir = os.path.join(self.root, "data")
        self.curated_data = os.path.join(self.data_dir, "squad", "kpi_train.json")
        # If True, curated data will be split by dev_split ratio to train and val and saved in train_filename,
        # dev_filename . Otherwise train and val data will be loaded from mentioned filenames.
        self.perform_splitting = settings["train_kpi"]["data"][
            "perform_splitting"
        ]  # was False initially
        self.seed = settings["train_kpi"]["seed"]
        self.dev_split = settings["train_kpi"]["data"]["dev_split"]
        self.train_filename = os.path.join(
            self.data_dir, "squad", "kpi_train_split.json"
        )
        self.dev_filename = os.path.join(self.data_dir, "squad", "kpi_val_split.json")
        self.test_filename = None
        self.saved_models_dir = os.path.join(self.root, "models", "KPI_EXTRACTION")
        self.dev_predictions_filename = os.path.join(
            self.root,
            "reports",
            "qa_predictions.json",
        )
        self.model_performance_metrics_filename = os.path.join(
            self.root,
            "reports",
            "qa_model_perf_metrics.csv",
        )
        self.num_processes = settings["train_kpi"]["training"]["max_processes"]

    def update_paths(self, data_dir, curated_data):
        """Update paths."""
        # self.data_dir = data_dir
        # self.curated_data = curated_data
        # self.train_filename = os.path.join(self.data_dir, f"train_split_{os.path.basename(self.curated_data)}")
        # self.dev_filename = os.path.join(self.data_dir, f"dev_split_{os.path.basename(self.curated_data)}")
        self.data_dir = os.path.join(self.root, "data")
        self.curated_data = os.path.join(self.data_dir, "squad", "kpi_train.json")
        self.train_filename = os.path.join(
            self.data_dir, "squad", "kpi_train_split.json"
        )
        self.dev_filename = os.path.join(self.data_dir, "squad", "kpi_val_split.json")


class QATokenizerConfig(QAConfig):
    """QATokenizerConfig class."""

    def __init__(self, project_name):
        """Initialize QATokenizerConfig class."""
        super().__init__(project_name)
        self.pretrained_model_name_or_path = base_lm_model
        self.do_lower_case = False


class QAProcessorConfig(QAConfig):
    """QAProcessorConfig class."""

    def __init__(self, project_name):
        """Initialize QAProcessorConfig class."""
        super().__init__(project_name)
        self.processor_name = "SquadProcessor"
        self.max_seq_len = settings["train_kpi"]["processor"]["max_seq_len"]
        self.label_list = settings["train_kpi"]["processor"]["label_list"]
        self.metric = settings["train_kpi"]["processor"]["metric"]


class QAModelConfig(QAConfig):
    """QAModelConfig class."""

    def __init__(self, project_name):
        """Initilize QAModelConfig class."""
        super().__init__(project_name)
        self.class_type = QuestionAnsweringHead
        self.head_config = {}

        # set to None if you don't want to load the config file for this model
        self.load_dir = None  # TODO: Should this really be None ?
        self.lang_model = base_lm_model
        self.layer_dims = settings["train_kpi"]["model"]["model_layer_dims"]
        self.lm_output_types = settings["train_kpi"]["model"]["model_lm_output_types"]


class QATrainingConfig(QAConfig):
    """QATrainingConfig class."""

    def __init__(self, project_name):
        """Initialize QATrainingConfig class."""
        super().__init__(project_name)
        self.run_hyp_tuning = settings["train_kpi"]["training"]["run_hyp_tuning"]
        self.use_cuda = True

        # Check if GPU exists
        if not torch.cuda.is_available():
            _logger.warning("No gpu available, setting use_cuda to False")
            self.use_cuda = False

        self.use_amp = settings["train_kpi"]["training"]["use_amp"]
        self.distributed = settings["train_kpi"]["training"]["distributed"]
        self.learning_rate = settings["train_kpi"]["training"]["learning_rate"]
        self.n_epochs = settings["train_kpi"]["training"]["n_epochs"]
        self.evaluate_every = settings["train_kpi"]["training"]["evaluate_every"]
        self.dropout = settings["train_kpi"]["training"]["dropout"]
        self.batch_size = settings["train_kpi"]["training"]["batch_size"]
        self.grad_acc_steps = settings["train_kpi"]["training"]["grad_acc_steps"]
        self.run_cv = settings["train_kpi"]["training"]["run_cv"]
        self.xval_folds = settings["train_kpi"]["training"]["xval_folds"]


class QAMLFlowConfig(QAConfig):
    """QAMLFlowConfig class."""

    def __init__(self, project_name):
        """Initialize QAMLFlowConfig class."""
        super().__init__(project_name)
        self.track_experiment = settings["train_kpi"]["mlflow"]["track_experiment"]
        self.run_name = self.experiment_name
        self.url = settings["train_kpi"]["mlflow"]["url"]


class QAInferConfig(QAConfig):
    """QAInferConfig class."""

    def __init__(self, project_name):
        """Initialize QAInferConfig class."""
        super().__init__(project_name)
        self.data_types = ["Text"]
        self.result_dir = {"Text": os.path.join(self.root, "data", "infer_kpi")}
        self.load_dir = {"Text": os.path.join(self.root, "models", "KPI_EXTRACTION")}
        self.relevance_dir = {
            "Text": os.path.join(self.root, "data", "infer_relevance")
        }

        self.skip_processed_files = settings["infer_kpi"]["skip_processed_files"]
        self.top_k = settings["infer_kpi"]["top_k"]
        self.batch_size = settings["infer_kpi"]["top_k"]
        self.gpu = settings["infer_kpi"]["gpu"]
        self.num_processes = settings["infer_kpi"]["num_processes"]
        self.no_ans_boost = settings["infer_kpi"]["no_ans_boost"]

        # Parameter for Table inference
        self.word_emb_type = "en_vectors_web_lg"
        self.fix_spelling = True
        self.regex_match_year = True
        self.years = (
            []
        )  # Set to a list of years to extract KPI for, otherwise the years are extracted from pdf name.
        self.num_years = 3  # Number of years to look for KPI answers, will be used if year is extracted from pdf name.
        self.extracted_dir = os.path.join(self.root, "data", "extraction", project_name)
