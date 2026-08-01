import cachetools
from cachetools import cached
import boto3
from botocore.client import Config
from PIL.Image import Image
from env import BUCKET_NAME, BUCKET_PATH
import os

# Initialize R2 client (Cloudflare R2 uses S3-compatible API)
# R2 endpoint format: https://<account_id>.r2.cloudflarestorage.com
# Credentials should be in ~/.aws/credentials or environment variables
s3_client = boto3.client(
    's3',
    endpoint_url=os.environ.get('R2_ENDPOINT_URL', 'https://your-account-id.r2.cloudflarestorage.com'),
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    config=Config(signature_version='s3v4'),
    region_name='auto'  # R2 uses 'auto' as the region
)

bucket_name = os.environ.get('R2_BUCKET_NAME', os.environ.get('CLOUDFLARE_BUCKET', BUCKET_NAME))
public_bucket_domain = os.environ.get('R2_PUBLIC_DOMAIN', BUCKET_NAME)
bucket_path = BUCKET_PATH  # static/uploads

@cached(cachetools.TTLCache(maxsize=10000, ttl=60 * 60 * 24))
def check_if_blob_exists(name: object) -> object:
    """Check if a file exists in the R2 bucket"""
    try:
        s3_client.head_object(Bucket=bucket_name, Key=get_name_with_path(name))
        return True
    except:
        return False

def upload_to_bucket(blob_name, path_to_file_on_local_disk, is_bytesio=False):
    """Upload data to R2 bucket"""
    key = get_name_with_path(blob_name)

    if not is_bytesio:
        # Upload from file
        s3_client.upload_file(path_to_file_on_local_disk, bucket_name, key)
    else:
        # Upload from bytes
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=path_to_file_on_local_disk,
            ContentType='image/webp'
        )

    # Return public URL
    # For R2 with custom domain, the URL format is: https://<custom-domain>/<path>
    return f"https://{public_bucket_domain}/{key}"


def get_name_with_path(blob_name):
    return bucket_path + '/' + blob_name
