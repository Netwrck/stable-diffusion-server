import os

# Storage provider can be 'r2' or 'gcs'. Default to R2 as it is generally
# cheaper and provides S3 compatible APIs.
STORAGE_PROVIDER = os.getenv("STORAGE_PROVIDER", "r2").lower()

# Name of the bucket to upload to. This bucket should be accessible via the
# public domain as configured in your cloud provider.
BUCKET_NAME = os.getenv("BUCKET_NAME", "netwrckstatic.netwrck.com")

# Path prefix inside the bucket where files are stored.
BUCKET_PATH = os.getenv("BUCKET_PATH", "static/uploads")

# Endpoint URL for R2/S3 compatible storage. For Cloudflare R2 this usually
# looks like `https://<accountid>.r2.cloudflarestorage.com`.
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "https://netwrckstatic.netwrck.com")

# Base public URL to prefix returned links. Defaults to the bucket name which
# assumes a custom domain is configured.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", BUCKET_NAME)
