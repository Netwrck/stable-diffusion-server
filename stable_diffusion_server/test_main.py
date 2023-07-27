from main import create_image_from_prompt

def test_create_image_from_prompt_sync():
    imagebytesresult = create_image_from_prompt("a test prompt")
    assert imagebytesresult is not None
