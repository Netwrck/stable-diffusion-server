import urllib.parse
from pathlib import Path

import nltk
from tqdm import tqdm

# from img2imgsdr import make_image
from scripts.ebank_api import get_image
from scripts.replace_special_chars import replace_special_chars

# import importlib
# importlib.set_lazy_imports(True)

# accept story text arg

topic = "backfill"

current_dir = Path(__file__).parent.absolute()
story_dir = current_dir / "stories" / topic
story_dir.mkdir(parents=True, exist_ok=True)
###https://image.netwrck.com/create_and_upload_image?prompt=beautiful%20scenery%20amazing%20land%20peaceful%20lowfi%20relaxing%20pathway%20anime&width=1920&height=1920&save_path=ai/beautiful-scenery-amazing-land-peaceful-lowfi-relaxing-pathway-tallestsq.webp

# df = pd.read_parquet(current_dir / './midjourney-prompts/data/test-00000-of-00001.parquet')
# df = pd.read_parquet(current_dir / './midjourney-prompts/data/train-00000-of-00001.parquet') #todo train
# print(df.size)

import re

characters_to_strip = '.,<>?/;:\'\"[]{}\\|`~!@#$%^&*()_+-='
# Create a translation table that maps each character in characters_to_strip to None
trans_table = str.maketrans(characters_to_strip, len(characters_to_strip) * ' ')


def remove_urlencoded_chars(prompt):
    prompt = prompt.replace("%22", "").replace("%20", " ").replace("%2C", ",").replace("%3A", ":").replace("%3F",
                                                                                                           "?").replace(
        "%2F", "/").replace("%3D", "=").replace("%26", "&").replace("%25", "%").replace("%3B", ";").replace("%2B",
                                                                                                            "+").replace(
        "%23", "#").replace("%3C", "<")
    # unurlencode
    prompt = urllib.parse.unquote(prompt)
    prompt = re.sub(r'[^\x00-\x7F]+', ' ', prompt)
    prompt = prompt.translate(trans_table)
    return prompt
# todo do this filtering in dataloading in search server

with open('ai.txt') as f:
    lines = f.readlines()

    # for i, sentence in tqdm(enumerate(df['text'])):
    #     if i<66011:
    #         continue
    for i, sentence in tqdm(enumerate(lines)):

        prompt = sentence.replace("\n", "").replace("gs://static.netwrck.com/static/uploads/ai", "").strip()
        # remove all the urlencodable bad chars e.g. %22
        prompt_clean = remove_urlencoded_chars(prompt)
        if prompt_clean == prompt:
            print("prompt is clean")
            print(prompt)
            continue
        prompt = prompt_clean
        print(prompt)
        # create prompt for creating backdrop
        #
        # backdrop_prompt = topic + " era backdrop environment for story, " + sentence
        ## cull all the special chars -- etc
        # remove all -- flags?
        # prompt = sentence.replace("--", "")
        try:
            if prompt.endswith(".webp"):
                prompt = prompt[:-5]
            if prompt.endswith(" webp"):
                prompt = prompt[:-5]
            if prompt.endswith("webp"):
                prompt = prompt[:-4]
            save_name = replace_special_chars(prompt.replace(" ", "-").replace("  ", " ")[:160])[:120]
            # save_path = f"ai/" + prompt.replace(" ", "-")
            # # request for backdrop from image.ebank.nz with prompt
            # backdrop_image = get_image(backdrop_prompt, save_path)
            # backdrop_image.save(str(story_dir / f"{i}backdrop.webp"), quality=85, optimize=True, format="webp")

            # create prompt for creating character
            # character_prompt = sentence + ' character foreground for story, ' + topic + " era"

            print(save_name)

            file_name = f"{save_name}.webp"
            if not (story_dir / file_name).exists():
                # if i % 2 == 0:
                backdrop_image = get_image(prompt, f"ai-{save_name}", width=1024, height=1024)
                continue

                # image = make_image(prompt)
                # image.save(str(story_dir / file_name), quality=85, optimize=True, format="webp")
        except Exception as e:
            print(e)
