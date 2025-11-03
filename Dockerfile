FROM nvidia/cuda:12.2.0-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends     git wget curl python3 python3-pip python3-dev build-essential libgl1 &&     rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/lightgan-ld
COPY requirements.txt ./
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt && pip3 install h5py pytest pre-commit

COPY . .
RUN pre-commit install-hooks || true

CMD ["bash", "-lc", "pytest -q && python -m src.cli.train --config configs/default.yaml"]
