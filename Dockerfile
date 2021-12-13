FROM dprl/scanssd-base:latest

COPY . .
RUN wget -q https://www.cs.rit.edu/~dprl/mathseer-pipeline/ssd512GTDB_256_epoch15.pth
RUN mv ssd512GTDB_256_epoch15.pth /workspace/src/trained_weights/
ENV PYTHONPATH="/workspace/"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR src/server
CMD ["conda", "run", "-n", "scanssd", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
