from io import BytesIO
import random
from sellerinfo import EBANK_API_KEY
from PIL import Image
import requests
from scripts.replace_special_chars import replace_special_chars

base_url_two = "https://images2.netwrck.com/create_and_upload_image?prompt="
base_url_one = "https://images.netwrck.com/create_and_upload_image?prompt="

base_urls= [base_url_two, base_url_one]
headers = {"secret": EBANK_API_KEY}

def get_random_base_url():
    return random.choice(base_urls)


def get_image(prompt, download_dir="", width=1080, height=1920, retries=3):
    base_url = get_random_base_url()
    if not download_dir:
        download_dir = replace_special_chars(prompt.replace(" ", "-"))
    request_url = base_url + prompt + f"&width={width}&height={height}&save_path=" + download_dir + ".webp"
    response = requests.get(request_url, headers=headers)
    try:
        json = response.json()
    except Exception as e:
        if retries > 0:
            print(e)
            print("response not json, retrying"
                  f"retries left: {retries}")
            return get_image(prompt, download_dir, width, height, retries-1)
        else:
            print(e)
            print(response)
            print(response.text)
            return None
    image_url = json["path"]
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image
