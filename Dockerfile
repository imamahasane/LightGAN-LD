FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /workspace/LightGAN-LD
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY pyproject.toml README.md requirements.txt ./
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements.txt && pip install -e .
COPY . .
CMD ["bash"]
