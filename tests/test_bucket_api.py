from io import BytesIO

from PIL import Image

import importlib
import os

import boto3
from moto import mock_aws

from stable_diffusion_server import bucket_api as bucket_api


@mock_aws
def test_upload_to_bucket():
    os.environ['STORAGE_PROVIDER'] = 'r2'
    os.environ['BUCKET_NAME'] = 'test-bucket'
    os.environ['BUCKET_PATH'] = 'static/uploads'
    os.environ['R2_ENDPOINT_URL'] = 'https://s3.amazonaws.com'

    import env
    importlib.reload(env)
    importlib.reload(bucket_api)
    bucket_api.init_storage()

    s3 = boto3.client('s3', endpoint_url=os.environ['R2_ENDPOINT_URL'])
    s3.create_bucket(Bucket='test-bucket')


    link = bucket_api.upload_to_bucket('test.txt', 'tests/test.txt')
    assert link == 'https://test-bucket/static/uploads/test.txt'
    assert bucket_api.check_if_blob_exists('test.txt')


@mock_aws
def test_upload_bytesio_to_bucket():
    os.environ['STORAGE_PROVIDER'] = 'r2'
    os.environ['BUCKET_NAME'] = 'test-bucket'
    os.environ['BUCKET_PATH'] = 'static/uploads'
    os.environ['R2_ENDPOINT_URL'] = 'https://s3.amazonaws.com'

    import env
    importlib.reload(env)
    importlib.reload(bucket_api)
    bucket_api.init_storage()

    s3 = boto3.client('s3', endpoint_url=os.environ['R2_ENDPOINT_URL'])
    s3.create_bucket(Bucket='test-bucket')

    pilimage = Image.open('tests/data/gunbladedraw.png').convert('RGB')
    bs = BytesIO()
    pilimage.save(bs, "jpeg")
    bio = bs.getvalue()
    link = bucket_api.upload_to_bucket('medi.png', bio, is_bytesio=True)
    assert link == 'https://test-bucket/static/uploads/medi.png'
    assert bucket_api.check_if_blob_exists('medi.png')
