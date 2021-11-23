from fastapi import FastAPI, File
from src.ssd import build_ssd
from src.data_loaders import exp_cfg
import torch.nn as nn
from pdf2image import convert_from_bytes
from src.server.prediction_service import predict
from src.args import parse_test_args, get_gpus
import asyncio


SCANSSD_CONF = "0.5"

app = FastAPI()
num_classes = 2  # +1 background
parser = parse_test_args()
devices = get_gpus()

args = parser.parse_args(["--model_type", "512",
                          "--cfg", "math_gtdb_512",
                          "--padding", [0, 2],
                          "--kernel", "1 5",
                          "--batch_size", str(devices[2]),
                          "--conf", SCANSSD_CONF,
                          "--window", "512"])

net = build_ssd(
    args, "test", exp_cfg[args.cfg], devices[0], args.model_type, num_classes
)
net = nn.DataParallel(net, device_ids=devices[0])

detect_queue = asyncio.Queue()


@app.get("/")
def read_root():
    return "Welcome to the deployed ScanSSD model, append '/docs' to this url to access the swagger web API"


@app.post("/predict")
def predict(dpi: int, file: bytes = File(...)):

    images = convert_from_bytes(pdf_file=file, dpi=dpi)
    predict(images)




