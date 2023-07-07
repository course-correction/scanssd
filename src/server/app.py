import zipfile
from io import StringIO, BytesIO
import copy
import numpy as np
import pandas as pd
from typing import List
from fastapi import FastAPI, File, Response, UploadFile
from typing_extensions import Annotated
from src.ssd import build_ssd
from src.data_loaders import exp_cfg
import torch.nn as nn
from pdf2image import convert_from_bytes
from src.server.prediction_service import predict_from_images
from src.args import parse_test_args, get_gpus
import torch
import uvicorn
from fastapi.responses import StreamingResponse
import sys
from PIL import Image

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

SCANSSD_CONF = "0.5"

app = FastAPI()
num_classes = 2  # +1 background
parser = parse_test_args()
gpu_data = get_gpus()

args = parser.parse_args(["--model_type", "512",
                          "--cfg", "math_gtdb_512",
                          "--padding", '0', '2',
                          "--kernel", '1', '5',
                          "--batch_size", str(gpu_data[2]),
                          "--conf", SCANSSD_CONF,
                          "--window", "512",
                          "--gpu", *gpu_data[0],
                          "--stride", "1.0",
                          "--post_process", "0",
                          "--op_mode", "pipeline",
                          "--trained_model", "../trained_weights/ssd512GTDB_256_epoch15.pth"])

gpus = [str(gpu_id) for gpu_id in args.gpu]
gpus = ','.join(gpus)
devices = torch.device('cuda:' + gpus)

net = build_ssd(
    args, "test", exp_cfg[args.cfg], devices, args.model_type, num_classes
)
net = nn.DataParallel(net, device_ids=args.gpu)

net.module.load_state_dict(
    torch.load(
        args.trained_model,
    )
)
net.eval()
gpus = [str(gpu_id) for gpu_id in args.gpu]
gpus = ','.join(gpus)
devices = torch.device('cuda:' + gpus)
net = net.to(devices)


@app.get("/")
def read_root():
    return "Welcome to the deployed ScanSSD inference model, append '/docs' to this url to access the swagger web API"


@app.post("/predict/")
def predict(dpi: int, conf: float, stride: float, file: bytes = File(...)):
    local_args = copy.deepcopy(args)
    local_args.conf = [conf]
    local_args.stride = stride
    images = convert_from_bytes(pdf_file=file, dpi=dpi)
    ret = predict_from_images(local_args, net, images)
    out_file_as_str = StringIO()
    np.savetxt(out_file_as_str, ret, fmt='%.2f', delimiter=',')
    response = StreamingResponse(
        iter([out_file_as_str.getvalue()]),
        media_type='text/csv',
        headers={
            'Content-Disposition': 'attachment;filename=predictions.csv',
            'Access-Control-Expose-Headers': 'Content-Disposition'
        }
    )
    # return
    return response


@app.post("/predict_pipeline/")
def predict_pipeline(conf: float, stride: float, images: List[UploadFile] = File(...)):
    local_args = copy.deepcopy(args)
    local_args.conf = [conf]
    images = sorted(images, key=lambda x: int(x.filename[:-4]))
    local_args.stride = stride
    images_pil = []
    for i in images:
        images_pil.append(Image.open(i.file))
    ret = predict_from_images(local_args, net, images_pil)
    out_file_as_str = StringIO()
    np.savetxt(out_file_as_str, ret, fmt='%.2f', delimiter=',')

    response = StreamingResponse(
        iter([out_file_as_str.getvalue()]),
        media_type='text/csv',
        headers={
            'Content-Disposition': 'attachment;filename=predictions.csv',
            'Access-Control-Expose-Headers': 'Content-Disposition'
        },
    )
    return response

@app.post("/convert")
async def convert(dpi: int, file: bytes = File(...)):
    images = convert_from_bytes(pdf_file=file, dpi=dpi)
    zip_io = BytesIO()
    zipped_images = zipfile.ZipFile(zip_io, "w")
    for i, img in enumerate(images):
        image_bytes = BytesIO()
        img.save(image_bytes, format=img.format)
        zipped_images.writestr(str(i) + ".png", image_bytes.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    zipped_images.close()
    resp = Response(zip_io.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={f"pdf2img_{dpi}.zip"}'
    })
    return resp


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
