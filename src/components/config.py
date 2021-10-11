import os
import src
import pathlib
import torch

# General config
STAGE = "extract"  # "extract" | "curate "
SEED = 42

ROOT = pathlib.Path(src.__file__).resolve().parent.parent
CONFIG_FOLDER = ROOT / "config"
CHECKPOINT_FOLDER = ROOT / "checkpoint"
# the data for demo notebooks is located at sample_data directory
DATA_FOLDER = ROOT / "data"
BASE_PDF_FOLDER = DATA_FOLDER / "pdfs"
BASE_ANNOTATION_FOLDER = DATA_FOLDER / "annotations"
BASE_EXTRACTION_FOLDER = DATA_FOLDER / "extraction"
BASE_CURATION_FOLDER = DATA_FOLDER / "curation"


# If installed as package, BASE_EXTRACTION_FOLDER will point to site-packages
# So we need to update how config params are passed
# if not os.path.exists(BASE_EXTRACTION_FOLDER):
#     os.mkdir(BASE_EXTRACTION_FOLDER)
# if not os.path.exists(BASE_CURATION_FOLDER):
#     os.mkdir(BASE_CURATION_FOLDER)

ckpt = "icdar_19b2_v2.pth" #if "cpu" in torch.__version__ else "icdar_19b2.pth"
config_file = (
    "cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py"
    # if "cpu" in torch.__version__
    # else "cascade_mask_rcnn_hrnetv2p_w32_20e.py"
)
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
    # Set to  ANNOTATION_FOLDER if you want to extract just pdfs mentioned in the annotations
    # Set to None to extract all pdfs in pdf folder (for production stage)
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

# Components
EXTRACTORS = [
    # ("PDFTableExtractor", PDFTableExtractor_kwargs),
    ("PDFTextExtractor", PDFTextExtractor_kwargs)
]

CURATORS = [
    ("TextCurator", TextCurator_kwargs)
    # ,("TableCurator", TableCurator_kwargs)
]
