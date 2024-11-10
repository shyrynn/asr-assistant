FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y wget git python3.10 python3-pip ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx ffmpeg\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install fastapi[standard] uvicorn[standard] Cython setuptools numpy pyannote.audio lmnr boto3 hydra-core
COPY requirements.txt .
RUN pip3 install -r requirements.txt 

RUN mkdir /app
COPY . /app

RUN chmod +x /app/src/download-weights.py
#RUN ["python3", "/app/src/download-weights.py"]

WORKDIR /app/src

ENTRYPOINT [ "bash", "run.sh" ]