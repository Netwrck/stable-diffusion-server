from stable_diffusion_server.bucket_api import upload_to_bucket


def test_upload_to_bucket():
    link = upload_to_bucket('test.txt', 'tests/test.txt')
    assert link == 'https://storage.googleapis.com/static.netwrck.com/static/uploads/test.txt'
    assert False
