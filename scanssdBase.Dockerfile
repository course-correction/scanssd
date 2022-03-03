# this is the base docker image for the scanssd image. it exist because it takes a really long time to build
# rebuilding it every time scanssd is rebuild would waste our gitlab ci/cd minutes
#
# docker build -f base.Dockerfile . -t dprl/scanssd-base:latest
# docker push dprl/scanssd-base:latest

FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install wget && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./niagarafonts.tar.gz ./
#RUN apt-get update && apt-get install unzip
RUN mkdir -p /usr/share/fonts/
RUN tar -xzvf niagarafonts.tar.gz -C /usr/share/fonts/

RUN conda create -n scanssd python=3.6.9
SHELL ["conda", "run", "-n", "scanssd", "/bin/bash", "-c"]
RUN pip install -r requirements.txt
RUN conda install -c conda-forge poppler
RUN conda update -n base -c defaults conda

CMD ["echo", "hello from scanssd base"]