"""Default runtime config."""
import pathlib
import os

# General config
STAGE = "extract"  # "extract" | "curate "
SEED = 42

if os.getenv("AUTOMATION"):
    ROOT = pathlib.Path("/opt/app-root")
else:
    ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

CONFIG_FOLDER = ROOT
CHECKPOINT_FOLDER = ROOT / "models"
DATA_FOLDER = ROOT / "data"
BASE_PDF_FOLDER = DATA_FOLDER / "pdfs"
BASE_ANNOTATION_FOLDER = DATA_FOLDER / "annotations"
BASE_EXTRACTION_FOLDER = DATA_FOLDER / "extraction"
BASE_CURATION_FOLDER = DATA_FOLDER / "curation"
BASE_PROCESSED_DATA = DATA_FOLDER / "processed"
BASE_INFER_KPI_FOLDER = DATA_FOLDER / "infer_KPI"
BASE_INFER_RELEVANCE_FOLDER = DATA_FOLDER / "infer_relevance"

EXPERIMENT_NAME = "test-demo-4"
SAMPLE_SIZE = "medium"


DATA_S3_PREFIX = f"{EXPERIMENT_NAME}/pipeline_run/{SAMPLE_SIZE}"
BASE_PDF_S3_PREFIX = f"{DATA_S3_PREFIX}/pdfs"
BASE_ANNOTATION_S3_PREFIX = f"{DATA_S3_PREFIX}/annotations"
BASE_EXTRACTION_S3_PREFIX = f"{DATA_S3_PREFIX}/extraction"
BASE_CURATION_S3_PREFIX = f"{DATA_S3_PREFIX}/curation"
BASE_INFER_RELEVANCE_S3_PREFIX = f"{DATA_S3_PREFIX}/infer_relevance"
BASE_INFER_KPI_S3_PREFIX = f"{DATA_S3_PREFIX}/infer_KPI"
BASE_INFER_KPI_TABLE_S3_PREFIX = f"{EXPERIMENT_NAME}/KPI_table"
BASE_SAVED_MODELS_S3_PREFIX = f"{DATA_S3_PREFIX}/saved_models"
# BASE_SAVED_MODELS_S3_PREFIX = "aicoe-osc-demo/saved_models"
# CHECKPOINT_S3_PREFIX = BASE_SAVED_MODELS_S3_PREFIX
CHECKPOINT_S3_PREFIX = "aicoe-osc-demo/saved_models"

ckpt = "icdar_19b2_v2.pth"
config_file = "cascade_mask_rcnn_hrnetv2p_w32_20e_v2.py"
PDFTableExtractor_kwargs = {
    "batch_size": -1,
    "cscdtabnet_config": CONFIG_FOLDER / config_file,
    "cscdtabnet_ckpt": CHECKPOINT_FOLDER / ckpt,
    "bbox_thres": 0.85,
    "dpi": 200,
}

# PDFTextExtractor
PDFTextExtractor_kwargs = {
    "min_paragraph_length": 30,
    "annotation_folder": None,
    "skip_extracted_files": False,
}

TableCurator_kwargs = {
    "neg_pos_ratio": 1,
    "create_neg_samples": True,
    "columns_to_read": [
        "company",
        "source_file",
        "source_page",
        "kpi_id",
        "year",
        "answer",
        "data_type",
    ],
    "company_to_exclude": ["CEZ"],
    "seed": SEED,
}

TextCurator_kwargs = {
    "retrieve_paragraph": False,
    "neg_pos_ratio": 1,
    "columns_to_read": [
        "company",
        "source_file",
        "source_page",
        "kpi_id",
        "year",
        "answer",
        "data_type",
        "relevant_paragraphs",
    ],
    "company_to_exclude": [],
    "create_neg_samples": True,
    "seed": SEED,
}

# config for KPI inference dataset curator
TRAIN_KPI_INFERENCE_COLUMNS_TO_READ = [
    "company",
    "source_file",
    "source_page",
    "kpi_id",
    "year",
    "answer",
    "data_type",
    "relevant_paragraphs",
]


class CurateConfig:
    """KPI inference data curation config."""

    def __init__(self):
        """Set default parameters for kpi text inference curation."""
        self.val_ratio = 0
        self.seed = SEED
        self.find_new_answerable = True
        self.create_unanswerable = True


# Text KPI Inference Curator
TextKPIInferenceCurator_kwargs = {
    "annotation_folder": BASE_ANNOTATION_FOLDER,
    "agg_annotation": BASE_ANNOTATION_FOLDER
    / "20201030 1Qbit aggregated_annotations_needs_correction.xlsx",
    "extracted_text_json_folder": BASE_EXTRACTION_FOLDER,
    "output_squad_folder": f"{DATA_FOLDER}/squad",
    "relevant_text_path": f"{BASE_INFER_RELEVANCE_FOLDER}/*.csv",
    # "relevant_text_path": DATA_FOLDER / "infer_relevance" / "text_3434.csv",
}

# Components
EXTRACTORS = [
    # ("PDFTableExtractor", PDFTableExtractor_kwargs),
    ("PDFTextExtractor", PDFTextExtractor_kwargs)
]

CURATORS = [
    ("TextCurator", TextCurator_kwargs)
    # ,("TableCurator", TableCurator_kwargs)
]
