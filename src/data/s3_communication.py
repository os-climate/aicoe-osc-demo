"""S3 communication tools."""
import os
import pathlib
import os.path as osp
import boto3
import pandas as pd
from enum import Enum
from io import BytesIO


class S3FileType(Enum):
    """Enum to describe possible file type upload/downloads that S3Communication can handle."""

    CSV = 0
    JSON = 1
    PARQUET = 2


class S3Communication(object):
    """
    Class to establish communication with a ceph s3 bucket.

    It connects with the bucket and provides methods to read and write data in parquet, csv, and json formats.
    """

    def __init__(
        self, s3_endpoint_url, aws_access_key_id, aws_secret_access_key, s3_bucket
    ):
        """Initialize communicator."""
        self.s3_endpoint_url = s3_endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_resource = boto3.resource(
            "s3",
            endpoint_url=self.s3_endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )
        self.bucket = s3_bucket

    def _upload_bytes(self, buffer_bytes, prefix, key):
        """Upload byte content in buffer to bucket."""
        s3_object = self.s3_resource.Object(self.bucket, osp.join(prefix, key))
        status = s3_object.put(Body=buffer_bytes)
        return status

    def _download_bytes(self, prefix, key):
        """Download byte content in bucket/prefix/key to buffer."""
        buffer = BytesIO()
        s3_object = self.s3_resource.Object(self.bucket, osp.join(prefix, key))
        s3_object.download_fileobj(buffer)
        return buffer.getvalue()

    def upload_file_to_s3(self, filepath, s3_prefix, s3_key):
        """Read file from disk and upload to s3 bucket/prefix/key."""
        with open(filepath, "rb") as f:
            status = self._upload_bytes(f.read(), s3_prefix, s3_key)
        return status

    def download_file_from_s3(self, filepath, s3_prefix, s3_key):
        """Download file from s3 bucket/prefix/key and save it to filepath on disk."""
        buffer_bytes = self._download_bytes(s3_prefix, s3_key)
        with open(filepath, "wb") as f:
            f.write(buffer_bytes)

    def upload_df_to_s3(
        self, df, s3_prefix, s3_key, filetype=S3FileType.PARQUET, **pd_to_ftype_args
    ):
        """
        Take as input the data frame to be uploaded, and the output s3_key.

        Then save the data frame in the defined s3 bucket.
        """
        buffer = BytesIO()
        if filetype == S3FileType.CSV:
            df.to_csv(buffer, **pd_to_ftype_args)
        elif filetype == S3FileType.JSON:
            df.to_json(buffer, **pd_to_ftype_args)
        elif filetype == S3FileType.PARQUET:
            df.to_parquet(buffer, **pd_to_ftype_args)
        else:
            raise ValueError(
                f"Received unexpected file type arg {filetype}. Can only be one of: {list(S3FileType)})"
            )

        status = self._upload_bytes(buffer.getvalue(), s3_prefix, s3_key)
        return status

    def download_df_from_s3(
        self, s3_prefix, s3_key, filetype=S3FileType.PARQUET, **pd_read_ftype_args
    ):
        """Read from s3 and see if the saved data is correct."""
        buffer_bytes = self._download_bytes(s3_prefix, s3_key)
        buffer = BytesIO(buffer_bytes)

        if filetype == S3FileType.CSV:
            df = pd.read_csv(buffer, **pd_read_ftype_args)
        elif filetype == S3FileType.JSON:
            df = pd.read_json(buffer, **pd_read_ftype_args)
        elif filetype == S3FileType.PARQUET:
            df = pd.read_parquet(buffer, **pd_read_ftype_args)
        else:
            raise ValueError(
                f"Received unexpected file type arg {filetype}. Can only be one of: {list(S3FileType)})"
            )
        return df

    def upload_files_in_dir_to_prefix(self, source_dir, s3_prefix):
        """
        Upload all files in a directory to under the s3 prefix, recursively.

        Excludes hidden files and directories by default.
        """
        # convert to pathlib path
        source_dir_pl = pathlib.Path(source_dir)

        # get all files and directories EXCEPT hidden ones
        upload_files_paths = list(source_dir_pl.rglob("[!.]*"))
        for fpath in upload_files_paths:
            self.upload_file_to_s3(fpath, s3_prefix, fpath.name)

    def download_files_in_prefix_to_dir(self, s3_prefix, destination_dir):
        """
        Download all files under a prefix to a directory.

        Modified from original code here: https://stackoverflow.com/a/33350380
        """
        paginator = self.s3_resource.meta.client.get_paginator("list_objects")
        for result in paginator.paginate(
            Bucket=self.bucket, Delimiter="/", Prefix=s3_prefix
        ):
            # download all files in the sub "directory", if any
            if result.get("CommonPrefixes") is not None:
                for subdir in result.get("CommonPrefixes"):
                    self.download_files_in_prefix_to_dir(
                        subdir.get("Prefix"),
                        destination_dir,
                    )
            # download files at the root of this prefix
            for file in result.get("Contents", []):
                dest_filename = osp.basename(file.get("Key"))
                if dest_filename:
                    dest_pathname = osp.join(destination_dir, dest_filename)
                    if not osp.exists(osp.dirname(dest_pathname)):
                        os.makedirs(osp.dirname(dest_pathname))
                    self.download_file_from_s3(dest_pathname, s3_prefix, dest_filename)
