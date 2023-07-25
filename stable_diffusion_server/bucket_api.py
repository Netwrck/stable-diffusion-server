from google.cloud import storage

from env import BUCKET_NAME, BUCKET_PATH

storage_client = storage.Client()
bucket_name = BUCKET_NAME  # Do not put 'gs://my_bucket_name'
bucket = storage_client.bucket(bucket_name)
bucket_path = BUCKET_PATH
def check_if_blob_exists(name):
    stats = storage.Blob(bucket=bucket, name=name).exists(storage_client)
    return stats

def upload_to_bucket(blob_name, path_to_file_on_local_disk):
    """ Upload data to a bucket"""
    blob = bucket.blob(bucket_path + blob_name)
    blob.upload_from_filename(path_to_file_on_local_disk)
    #returns a public url
    return blob.public_url
