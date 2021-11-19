from fastapi import FastAPI, File
from src.ssd import build_ssd
from src.data_loaders import exp_cfg
import torch.nn as nn
from pdf2image import convert_from_bytes
import PIL
import numpy
from src.args import parse_test_args, get_gpus
import asyncio

app = FastAPI()
num_classes = 2  # +1 background
parser = parse_test_args()
args = parser.parse_args()
devices = get_gpus()

# TODO we need [kernel, padding, conf]
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

    pil_image = PIL.Image.open('Image.jpg').convert('RGB')
    open_cv_image = numpy.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()




