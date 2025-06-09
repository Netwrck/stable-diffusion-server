import os

BUCKET_NAME = 'static.netwrck.com'
BUCKET_PATH = 'static/uploads'

# Default Flux pipeline model name. Can be overridden with the
# ``FLUX_MODEL_NAME`` environment variable.
FLUX_MODEL_NAME = os.environ.get(
    "FLUX_MODEL_NAME", "black-forest-labs/FLUX.1-schnell"
)

# Toggle loading of the Flux pipeline via the ``ENABLE_FLUX_PIPELINE``
# environment variable. Any value other than "0" enables it.
ENABLE_FLUX_PIPELINE = os.environ.get("ENABLE_FLUX_PIPELINE", "1") != "0"
