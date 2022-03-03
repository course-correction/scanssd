# ScanSSD Server

This document is a supplement to the Main Readme provided in this repo. It describes how ScanSSD
may be deployed as a server which can accept PDFs, and return information about their bounding boxes.

It is built on that work and all acknowledgements and citations listed in the main Readme 
apply here as well. 

## Installation
### Requirements

 * Python
 * Conda package management system (Python)
 * (Optional) Docker Nvidia
   * Instructions for installation can be found https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide


#### Python Web Service (with FastAPI)

After following the Python installation instructions in the main readme you should 
have a Conda environment with all the necessary dependencies. Follow the directions below to start the server
```shell
make start-server
```
or 
```shell
$ export PYTHONPATH="${PYTHONPATH}:${PWD}"
$ conda activate scanssd
(scanssd) $ cd src/server
(scanssd) $ uvicorn app:app 
```
This will run the FastAPI server on port 8000.

At this point, you will see the server start up and produce this output:

```
INFO:     Started server process [233635]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

You can then go to the swagger docs URL ([http://localhost:8000/docs](http://localhost:8000/docs)) 
to interact with the server (e.g., passing PDF files to check CSV output).  

Requests can also be made from the command line using Curl commands, for example issuing the 
following command in a new terminal window:

```shell
curl -X 'POST' \
   'http://0.0.0.0:8000/predict/?dpi=256&conf=0.5&stride=0.75' \
   -H 'accept: application/json' \
   -H 'Content-Type: multipart/form-data' \
   -F 'file=@quick_start_data/pdf/Emden76.pdf;type=application/pdf'
```
will run ScanSSD-XYc over a test page image, and return a list of page numbers and 
bounding boxes at the terminal. All values are floating point numbers (including
the first page number '0' as '0.00').

`ctrl-c` should stop the service in linux/unix shells.

### Testing Server

After starting the server using `make start-server` as above, to test the
server, open a **new** terminal window, and issue the following command:

```shell
make test-server
```

This will run the local (command-line) and server on the same input files,
and then run `diff` on the resulting detection results. Differences seen
at the terminal should be minor rounding differences in coordinates between
the local and server outputs. The script `bin/ssdserver-test` is used to
run the server tests (this is created on installation).

### Docker Installation

If you have [nvidia docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed and running on your machine, issue:

```shell script
$ docker run --gpus all -p 8000:80 dprl/scanssd:latest
```

The system is then ready to process requests. 

After installing the docker container as described above, point your web browser to [http://localhost:8000/docs](http://localhost:8000/docs). This is the same interface that the FastAPI server creates, as the docker container essentially wraps the FastAPI server.

**Development Note:** This note is for those looking to modify the docker container for their own purposes (the one provided should work as-is, most people reading can skip this).

From the top level directory of ScanSSD, to rebuild the docker container on your machine, make sure to use a tag (&lt;sometag&gt;) other than `latest`. You can use this command to build a new  docker image:

```shell script
docker build -t scanssd:<sometag> . 
```
where `.` indicates that `Dockerfile` in the current directory should be used to specify the new docker container environment and parameters.



