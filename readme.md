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
git clone https://huggingface.co/segmind/SSD-1B
git clone https://huggingface.co/latent-consistency/lcm-ssd-1b 

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

http://localhost:8000/create_and_upload_image?prompt=good%20looking%20elf%20fantasy%20character&save_path=created/elf.webp

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
