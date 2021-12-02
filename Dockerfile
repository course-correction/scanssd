FROM dprl/scanssd-base

COPY . .

ENV PYTHONPATH="/workspace/"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN wget -q https://www.cs.rit.edu/~dprl/mathseer-pipeline/ssd512GTDB_256_epoch15.pth
RUN mv ssd512GTDB_256_epoch15.pth src/trained_weights/
# export path
# add utf 8 stuff
WORKDIR src/server

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]