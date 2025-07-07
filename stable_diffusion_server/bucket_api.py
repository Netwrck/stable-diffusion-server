"""Abstraction layer for cloud bucket uploads.

This module supports both Google Cloud Storage (GCS) and any S3 compatible
storage such as Cloudflare R2. The storage backend is selected via the
``STORAGE_PROVIDER`` environment variable defined in :mod:`env`.
"""

import cachetools
from cachetools import cached
from PIL.Image import Image

from env import (
    STORAGE_PROVIDER,
    BUCKET_NAME,
    BUCKET_PATH,
    R2_ENDPOINT_URL,
    PUBLIC_BASE_URL,
)

storage_client = None
bucket = None
s3_client = None
bucket_name = BUCKET_NAME
bucket_path = BUCKET_PATH


def init_storage():
    """Initialise global storage clients based on environment variables."""
    global storage_client, bucket, s3_client, bucket_name, bucket_path
    bucket_name = BUCKET_NAME
    bucket_path = BUCKET_PATH

    if STORAGE_PROVIDER == "gcs":
        from google.cloud import storage

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
    else:
        import boto3

        session = boto3.session.Session()
        s3_client = session.client("s3", endpoint_url=R2_ENDPOINT_URL)


init_storage()

@cached(cachetools.TTLCache(maxsize=10000, ttl=60 * 60 * 24))
def check_if_blob_exists(name: object) -> object:
    if STORAGE_PROVIDER == "gcs":
        stats = storage.Blob(bucket=bucket, name=get_name_with_path(name)).exists(storage_client)
        return stats
    else:
        try:
            s3_client.head_object(Bucket=bucket_name, Key=get_name_with_path(name))
            return True
        except s3_client.exceptions.ClientError as exc:  # type: ignore[attr-defined]
            if exc.response.get("Error", {}).get("Code") == "404":
                return False
            raise

def upload_to_bucket(blob_name, path_to_file_on_local_disk, is_bytesio=False):
    """Upload data to a bucket and return the public URL."""
    key = get_name_with_path(blob_name)
    if STORAGE_PROVIDER == "gcs":
        blob = bucket.blob(key)
        if not is_bytesio:
            blob.upload_from_filename(path_to_file_on_local_disk)
        else:
            blob.upload_from_string(path_to_file_on_local_disk, content_type="image/webp")
        return blob.public_url
    else:
        if not is_bytesio:
            s3_client.upload_file(path_to_file_on_local_disk, bucket_name, key, ExtraArgs={"ACL": "public-read"})
        else:
            s3_client.put_object(Bucket=bucket_name, Key=key, Body=path_to_file_on_local_disk, ACL="public-read", ContentType="image/webp")
        return f"https://{PUBLIC_BASE_URL}/{key}"


def get_name_with_path(blob_name):
    return bucket_path + '/' + blob_name
