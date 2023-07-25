simple stable diffusion server that saves images to cloud storage - returns links to google cloud storage

## Creators
[![netwrck logo](https://static.netwrck.com/static/img/netwrck-logo-colord256.png)](https://netwrck.com)

Checkout [AI Characters to chat with](https://netwrck.com) at [netwrck.com](https://netwrck.com)

Characters are narrated and written by many GPT models trained on 1000s of fantasy novels and chats.

Also for LLMs for making Text - Checkout [Text-Generator.io](https://text-generator.io) for a Open Source text generator that uses many AI models to generate the best along with image understanding and OCR networks.
## Setup

. Create a virtual environment (optional)

```bash
python3 -m venv venv
source venv/bin/activate
```

. Install dependencies

```bash
pip install -r requirements.txt
pip install -r dev-requirements.txt

cd models
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9    
```

. Edit settings in env.py
. download your Google cloud credentials to secrets/google-credentials.json
. Run the server

```bash
GOOGLE_APPLICATION_CREDENTIALS=secrets/google-credentials.json gunicorn  -k uvicorn.workers.UvicornWorker -b :8000 main:app --timeout 600 -w 1 
```

. Make a Request

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

