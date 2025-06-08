install:
	pip install uv
	uv pip install -r requirements.txt -r dev-requirements.txt


test:
	pytest -q


lint:
	flake8


docker-runpod:
	docker buildx build --tag runpod-image .


docker-cloudrun:
	docker buildx build --tag cloudrun-image .
