FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

COPY . .

RUN pip install -r requirements.txt

WORKDIR src/server

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]