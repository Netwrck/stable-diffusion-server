import argparse
from stable_diffusion_server.bucket_api import upload_to_bucket

parser = argparse.ArgumentParser(description="Upload a file to the configured bucket")
parser.add_argument("source", help="Local file path")
parser.add_argument("dest", help="Destination key inside bucket")
parser.add_argument("--bytes", action="store_true", help="Treat source as BytesIO")

args = parser.parse_args()

if args.bytes:
    with open(args.source, "rb") as f:
        data = f.read()
    url = upload_to_bucket(args.dest, data, is_bytesio=True)
else:
    url = upload_to_bucket(args.dest, args.source)
print(url)
