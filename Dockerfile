FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/solaris-main

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY requirements.txt requirements_gpu.txt setup.py ./
COPY src ./src
COPY config ./config
COPY assets ./assets
COPY static ./static
COPY vlm_eval ./vlm_eval
COPY vpt_dataset ./vpt_dataset
COPY unshard_dataset.py ./

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements_gpu.txt
RUN pip install -e .

CMD ["/bin/bash"]