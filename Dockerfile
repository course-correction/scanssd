FROM dprl/scanssd-base

COPY . .

ENV PYTHONPATH="/workspace/"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# export path
# add utf 8 stuff
WORKDIR src/server

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]