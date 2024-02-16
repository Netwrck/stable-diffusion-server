from pathlib import Path

import nltk
import requests

from img2imgsdr import make_image
from scripts.ebank_api import get_image

# accept story text arg

topic = "Queen Elizabeth I"
story = """Queen Elizabeth I was one of the most powerful and influential monarchs in history, ruling England for 44 years from 1558 until her death in 1603. But did you know these fun facts about her?

1. She was a redhead: Elizabeth was known for her fiery red hair, which she inherited from her mother, Anne Boleyn. She often wore elaborate wigs to enhance her natural locks.

2. She was highly educated: Elizabeth was fluent in six languages, including Latin, Greek, French, and Italian. She was also well-versed in history, philosophy, and theology.

3. She never married: Despite numerous suitors and proposals, Elizabeth never married. She famously declared herself married to her country, earning her the nickname "The Virgin Queen."

4. She loved to dance: Elizabeth was an accomplished dancer and often participated in court dances. She even danced with Sir Francis Drake after he successfully circumnavigated the globe.

5. She survived numerous assassination attempts: Elizabeth faced several attempts on her life, including the famous Babington Plot in 1586. She was able to outsmart her would-be assassins and maintain her hold on the throne.

These are just a few of the many fascinating facts about Queen Elizabeth I. Her legacy continues to inspire and captivate people today."""

sentences = nltk.sent_tokenize(story)

current_dir = Path(__file__).parent.absolute()
story_dir = current_dir / "stories" / topic
story_dir.mkdir(parents=True, exist_ok=True)
###https://image.netwrck.com/create_and_upload_image?prompt=beautiful%20scenery%20amazing%20land%20peaceful%20lowfi%20relaxing%20pathway%20anime&width=1920&height=1920&save_path=ai/beautiful-scenery-amazing-land-peaceful-lowfi-relaxing-pathway-tallestsq.webp

for i, sentence in enumerate(sentences):
    # create prompt for creating backdrop
    #
    backdrop_prompt = topic + " era backdrop environment for story, " + sentence
    save_path = f"ai/" + backdrop_prompt.replace(" ", "-")
    # request for backdrop from image.ebank.nz with prompt
    backdrop_image = get_image(backdrop_prompt, save_path)
    backdrop_image.save(str(story_dir / f"{i}backdrop.webp"), quality=85, optimize=True, format="webp")

    # create prompt for creating character
    character_prompt = sentence + ' character foreground for story, ' + topic + " era"
    image = make_image(character_prompt)
    image.save(str(story_dir / f"{i}char.webp"), quality=85, optimize=True, format="webp")

