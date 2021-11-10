import logging
import os

import torch
from farm.modeling.prediction_head import QuestionAnsweringHead

from config_farm_train import Config

_logger = logging.getLogger(__name__)

# E.g. bert-base-uncased, roberta-base, roberta-large, albert-base-v2, albert-large-v2
# or any hugging face model e.g. deepset/roberta-base-squad2, a-ware/roberta-large-squadv2
# Full list at huggingface.co/models
base_LM_model = "a-ware/roberta-large-squadv2"


class QAConfig(Config):
    def __init__(self, project_name):
        super().__init__(experiment_type="KPI_EXTRACTION", project_name=project_name)
        self.experiment_name = project_name
        self.data_type = "Text"


class QAFileConfig(QAConfig):

    def __init__(self, project_name):
        super().__init__(project_name)
        self.data_dir = os.path.join(self.root, "data")
        self.curated_data = os.path.join(self.data_dir, "training", project_name, "kpi_train.json")
        # If True, curated data will be split by dev_split ratio to train and val and saved in train_filename,
        # dev_filename . Otherwise train and val data will be loaded from mentioned filenames.
        self.perform_splitting = True #was False initially
        self.seed = 42
        self.dev_split = .2
        self.train_filename = os.path.join(self.data_dir, "training", project_name, "kpi_train_split.json")
        self.dev_filename = os.path.join(self.data_dir, "training", project_name, "kpi_val_split.json")
        self.test_filename = None
        self.saved_models_dir = os.path.join(self.root, "models", "KPI_EXTRACTION")

    def update_paths(self, data_dir, curated_data):
        #self.data_dir = data_dir
        #self.curated_data = curated_data
        #self.train_filename = os.path.join(self.data_dir, f"train_split_{os.path.basename(self.curated_data)}")
        #self.dev_filename = os.path.join(self.data_dir, f"dev_split_{os.path.basename(self.curated_data)}")
        self.data_dir = os.path.join(self.root, "data")
        self.curated_data = os.path.join(self.data_dir, "training", project_name, "kpi_train.json")
        self.train_filename = os.path.join(self.data_dir, "training", project_name, "kpi_train_split.json")
        self.dev_filename = os.path.join(self.data_dir, "training", project_name, "kpi_val_split.json")


class QATokenizerConfig(QAConfig):

    def __init__(self, project_name):
        super().__init__(project_name)
        self.pretrained_model_name_or_path = base_LM_model
        self.do_lower_case = False


class QAProcessorConfig(QAConfig):

    def __init__(self, project_name):
        super().__init__(project_name)
        self.processor_name = "SquadProcessor"
        self.max_seq_len = 384
        self.label_list = ["start_token", "end_token"]
        self.metric = "squad"


class QAModelConfig(QAConfig):

    def __init__(self, project_name):
        super().__init__(project_name)
        self.class_type = QuestionAnsweringHead
        self.head_config = {}

        # set to None if you don't want to load the config file for this model
        self.load_dir = None #TODO: Should this really be None ?
        self.lang_model = base_LM_model
        self.layer_dims = [768, 2]
        self.lm_output_types = ["per_token"]


class QATrainingConfig(QAConfig):

    def __init__(self, project_name):
        super().__init__(project_name)
        self.run_hyp_tuning = False
        self.use_cuda = True

        # Check if GPU exists
        if not torch.cuda.is_available():
            _logger.warning("No gpu available, setting use_cuda to False")
            self.use_cuda = False

        self.use_amp = True
        self.distributed = False
        self.learning_rate = 2e-5
        self.n_epochs = 5
        self.evaluate_every = 50
        self.dropout = 0.1
        self.batch_size = 4
        self.grad_acc_steps = 8
        self.run_cv = False  # running cross-validation won't save a model
        if self.run_cv:
            self.evaluate_every = 0
        self.xval_folds = 5


class QAMLFlowConfig(QAConfig):

    def __init__(self, project_name):
        super().__init__(project_name)
        self.track_experiment = False
        self.run_name = self.experiment_name
        self.url = "http://localhost:5000"


class QAInferConfig(QAConfig):

    def __init__(self, project_name):
        super().__init__(project_name)
        # please change the following accordingly
        #########Common parameters for Text and Table
        self.data_types = ["Text"]
        self.skip_processed_files = False  # If set to True, will skip inferring on already processed files
        self.top_k = 4
        self.result_dir = {"Text": os.path.join(self.root, "data", "infer_kpi")}
                           #"Table": os.path.join(self.root, "data", "infer", self.experiment_type, self.experiment_name, "Table")}

        # Parameters for text inference
        self.load_dir = {"Text": os.path.join(self.root, "models", "KPI_EXTRACTION")}
        self.batch_size = 16
        self.gpu = True
        # Set to value 1 (or 0) to disable multiprocessing. Set to None to let Inferencer use all CPU cores minus one.
        self.num_processes = None
        self.no_ans_boost = -15 # If increased, this will boost "No Answer" as prediction.
        # use large negative values (like -100) to disable giving "No answer" option.
        self.relevance_dir = {"Text": os.path.join(self.root, "data", "infer_relevance")}


        # Parameter for Table inference
        self.word_emb_type = "en_vectors_web_lg"
        self.fix_spelling = True
        self.regex_match_year = True
        self.years = [] # Set to a list of years to extract KPI for, otherwise the years are extracted from pdf name.
        self.num_years = 3 # Number of years to look for KPI answers, will be used if year is extracted from pdf name.
        self.extracted_dir = os.path.join(self.root, "data", "extraction", project_name)


