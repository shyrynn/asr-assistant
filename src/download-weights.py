import boto3
from loguru import logger
import os
import tarfile

access_key = os.environ["ACCESS_KEY"]
secret_key = os.environ["SECRET_KEY"]

host = os.environ["HOST"]
bucket_name = "Models"

s3 = boto3.client(
    "s3",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    endpoint_url=host,
)

def download_weight():
    keyname = "asr/giga/rnnt_model_weights.ckpt"
    local_path = "/app/rnnt_model_weights.ckpt"
    logger.info(f"Downloading weights : {local_path}")
    s3.download_file(bucket_name, keyname, local_path)
    
def download_tokenizer():
    keyname = "asr/giga/tokenizer_all_sets.tar"
    local_path = "/app/tokenizer_all_sets.tar"
    logger.info(f"Downloading weights : {local_path}")
    s3.download_file(bucket_name, keyname, local_path)
    
def untar(path):
    with tarfile.open(path, 'r:*') as tar:
        tar.extractall(path='.')

    
def download_all():
    download_weight()
    download_tokenizer()
    untar('/app/tokenizer_all_sets.tar')

if __name__ == "__main__":
    download_all()