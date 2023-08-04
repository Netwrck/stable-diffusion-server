simple stable diffusion server that saves images to cloud storage - returns links to google cloud storage

## Creators
[![netwrck logo](https://static.netwrck.com/static/img/netwrck-logo-colord256.png)](https://netwrck.com)

Checkout [Voiced AI Characters to chat with](https://netwrck.com) at [netwrck.com](https://netwrck.com)

Characters are narrated and written by many GPT models trained on 1000s of fantasy novels and chats.

Also for LLMs for making Text - Checkout [Text-Generator.io](https://text-generator.io) for a Open Source text generator that uses many AI models to generate the best along with image understanding and OCR networks.
## Setup

. Create a virtual environment (optional)

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Install dependencies

```bash
pip install -r requirements.txt
pip install -r dev-requirements.txt

cd models
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0    

# install stopwords
python -c "import nltk; nltk.download('stopwords')"
```

#### Edit settings in env.py
#### download your Google cloud credentials to secrets/google-credentials.json
Images generated will be stored in your bucket
#### Run the server

```bash
GOOGLE_APPLICATION_CREDENTIALS=secrets/google-credentials.json gunicorn  -k uvicorn.workers.UvicornWorker -b :8000 main:app --timeout 600 -w 1 
```

with max 4 requests at a time
This will drop a lot of requests under load instead of taking on too much work and causing OOM Errors.

```bash
GOOGLE_APPLICATION_CREDENTIALS=secrets/google-credentials.json PYTHONPATH=. uvicorn --port 8000 --timeout-keep-alive 600 --workers 1 --backlog 1 --limit-concurrency 4 main:app
```

#### Make a Request

```bash
http://localhost:8000/create_and_upload_image?prompt=good%20looking%20elf%20fantasy%20character&save_path=created/elf.png
```
Response
```shell
{"path":"https://storage.googleapis.com/static.netwrck.com/static/uploads/created/elf.png"}
```

http://localhost:8000/docs


Check to see that "good Looking elf fantasy character" was created

![elf.png](https://storage.googleapis.com/static.netwrck.com/static/uploads/created/elf.png)
![elf2.png](https://storage.googleapis.com/static.netwrck.com/static/uploads/created/elf2.png)

### Testing

```bash
GOOGLE_APPLICATION_CREDENTIALS=secrets/google-credentials.json pytest .
```

