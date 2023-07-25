from io import BytesIO

from PIL import Image

from stable_diffusion_server.bucket_api import upload_to_bucket, check_if_blob_exists


def test_upload_to_bucket():
    link = upload_to_bucket('test.txt', 'tests/test.txt')
    assert link == 'https://storage.googleapis.com/static.netwrck.com/static/uploads/test.txt'
    #  check if file exists
    assert check_if_blob_exists('test.txt')

def test_upload_bytesio_to_bucket():
    # bytesio = open('backdrops/medi.png', 'rb')
    pilimage = Image.open('backdrops/medi.png')
    # bytesio = pilimage.tobytes()
    bs = BytesIO()
    pilimage.save(bs, "jpeg")
    bio = bs.getvalue()
    link = upload_to_bucket('medi.png', bio, is_bytesio=True)
    assert link == 'https://storage.googleapis.com/static.netwrck.com/static/uploads/medi.png'
    #  check if file exists
    assert check_if_blob_exists('medi.png')
