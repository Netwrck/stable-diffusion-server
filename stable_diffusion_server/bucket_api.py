import cachetools
from cachetools import cached
from google.cloud import storage
from PIL.Image import Image
from env import BUCKET_NAME, BUCKET_PATH

storage_client = storage.Client()
bucket_name = BUCKET_NAME  # Do not put 'gs://my_bucket_name'
bucket = storage_client.bucket(bucket_name)
bucket_path = BUCKET_PATH

@cached(cachetools.TTLCache(maxsize=10000, ttl=60 * 60 * 24))
def check_if_blob_exists(name: object) -> object:
    stats = storage.Blob(bucket=bucket, name=get_name_with_path(name)).exists(storage_client)
    return stats

def upload_to_bucket(blob_name, path_to_file_on_local_disk, is_bytesio=False):
    """ Upload data to a bucket"""
    blob = bucket.blob(get_name_with_path(blob_name))
    if not is_bytesio:
        blob.upload_from_filename(path_to_file_on_local_disk)
    else:
        blob.upload_from_string(path_to_file_on_local_disk, content_type='image/webp')
    #returns a public url
    return blob.public_url


def get_name_with_path(blob_name):
    return bucket_path + '/' + blob_name
