from io import BytesIO

from PIL import Image
import requests

base_url = "https://image.netwrck.com/create_and_upload_image?prompt="
def get_image(prompt, download_dir):
    response = requests.get(base_url + prompt + "&width=1080&height=1920&save_path=" + download_dir + '.webp')
    json = response.json()
    image_url = json['path']
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image
