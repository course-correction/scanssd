FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y poppler-utils
RUN pip install -r requirements.txt

CMD ["echo", "hello from scanssd base"]