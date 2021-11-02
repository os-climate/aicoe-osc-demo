"""Load kpi mapping."""
import os
import pathlib
from dotenv import load_dotenv
from src.data.s3_communication import S3FileType, S3Communication

# Load credentials
dotenv_dir = os.environ.get(
    "CREDENTIAL_DOTENV_DIR", os.environ.get("PWD", "/opt/app-root/src")
)
dotenv_path = pathlib.Path(dotenv_dir) / "credentials.env"
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)

# Init s3 communication
s3c = S3Communication(
    s3_endpoint_url=os.getenv("S3_ENDPOINT"),
    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
    s3_bucket=os.getenv("S3_BUCKET"),
)

# Read kpi mapping csv from s3
df = s3c.download_df_from_s3(
    "corpdata/ESG/kpi_mapping",
    "kpi_mapping.csv",
    filetype=S3FileType.CSV,
    header=0,
)

_KPI_MAPPING = {str(i[0]): i[1] for i in df[["kpi_id", "question"]].values}
KPI_MAPPING = {(float(key)): value for key, value in _KPI_MAPPING.items()}

_KPI_MAPPING_MODEL = {
    str(i[0]): (i[1], [j.strip() for j in i[2].split(',')]) \
    for i in df[['kpi_id', 'question', 'sectors']].values
}
KPI_MAPPING_MODEL = {(float(key)): value for key, value in _KPI_MAPPING_MODEL.items()}

# Which questions should be added the year
ADD_YEAR = df[df["add_year"]].kpi_id.tolist()

# Category where the answer to the question should originate from
KPI_CATEGORY = {
    i[0]: [j.strip() for j in i[1].split(", ")]
    for i in df[["kpi_id", "kpi_category"]].values
}
