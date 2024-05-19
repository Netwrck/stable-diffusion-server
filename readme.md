Simple Stable Diffusion Server 

![art stable diffusion server logo](https://static.netwrck.com/static/uploads/aiartstation-art-server-logo-minimalist-artistic-computer-stable-diffusion-art-server-company-confident-engaging-wow-3.webp)

Can run locally for style transfer art generation and inpainting.

Can be ran in production mode that saves images to cloud storage - returns links to google cloud storage

For Hosted AI Art Generation checkout [AI Art Generator and Search Engine](https://aiart-generator.art) which also makes videos and 2k upscaled images.

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
git clone git@hf.co:/stabilityai/stable-diffusion-xl-base-1.0
git clone git@hf.co:/dataautogpt3/ProteusV0.2

# optional models for style transfer
git clone git@hf.co:/diffusers/controlnet-canny-sdxl-1.0

# older models you could try

#git clone git@hf.co:/latent-consistency/lcm-ssd-1b 

# install stopwords
python -c "import nltk; nltk.download('stopwords')"
```

#### Run the Gradio UI

```
python gradio_ui.py
```
Go to 
http://127.0.0.1:7860


![gradio demo](gradioimg.png)

## Server setup
#### Edit settings
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

http://localhost:8000/create_and_upload_image?prompt=good%20looking%20elf%20fantasy%20character&save_path=created/elf.webp

Response
```shell
{"path":"https://storage.googleapis.com/static.netwrck.com/static/uploads/created/elf.png"}
```

http://localhost:8000/swagger-docs


Check to see that "good Looking elf fantasy character" was created

![elf.png](https://storage.googleapis.com/static.netwrck.com/static/uploads/aiamazing-good-looking-elf-fantasy-character-awesome-portrait-2.webp)
![elf2.png](https://github.com/Netwrck/stable-diffusion-server/assets/2122616/81e86fb7-0419-4003-a67a-21470df225ea)

### Testing

```bash
GOOGLE_APPLICATION_CREDENTIALS=secrets/google-credentials.json pytest .
```


#### Running under supervisord

edit ops/supervisor.conf

install the supervisor
apt-get install -y supervisor
```bash
sudo cat >/etc/supervisor/conf.d/python-app.conf << EOF
[program:sdif_http_server]
directory=/home/lee/code/sdif
command=/home/lee/code/sdif/.env/bin/uvicorn --port 8000 --timeout-keep-alive 600 --workers 1 --backlog 1 --limit-concurrency 4 main:app
autostart=true
autorestart=true
environment=VIRTUAL_ENV="/home/lee/code/sdif/.env/",PATH="/opt/app/sdif/.env/bin",HOME="/home/lee",GOOGLE_APPLICATION_CREDENTIALS="secrets/google-credentials.json",PYTHONPATH="/home/lee/code/sdif"
stdout_logfile=syslog
stderr_logfile=syslog
user=lee
EOF

supervisorctl reread
supervisorctl update
```

#### run a manager process to kill/restart if the server if it is hanging

Sometimes the server just stops working and needs a hard restart

This command will kill the server if it is hanging and restart it (must be running under supervisorctl)
```
python3 manager.py
```

# hack restarting without supervisor
run the server in a infinite loop
```
while true; do GOOGLE_APPLICATION_CREDENTIALS=secrets/google-credentials.json PYTHONPATH=. uvicorn --port 8000 --timeout-keep-alive 600 --workers 1 --backlog 1 --limit-concurrency 4 main:app; done
```

# windows setup

py -3.11 -m venv .wvenv
. .wvenv/Scripts/activate
python -m pip install -r requirements.txt

# contributing guidelines
Please help in any way.


## Shameless Plug from Maintainers
[![netwrck logo](https://static.netwrck.com/static/img/netwrck-logo-colord256.png)](https://netwrck.com)
[![eBank logo](https://static.netwrck.com/static/img/ebank-logo-removebg-full387.png)](https://aiart-generator.art)

Checkout [Voiced AI Characters to chat with](https://netwrck.com) at [netwrck.com](https://netwrck.com)

Characters are narrated and written by many GPT models trained on 1000s of fantasy novels and chats.

For Vision LLMs for making Text - Checkout [Text-Generator.io](https://text-generator.io) for a Open Source text generator that uses many AI models to generate the best along with image understanding and OCR networks.