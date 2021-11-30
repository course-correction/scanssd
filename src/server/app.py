from io import StringIO
import copy
import pandas as pd
from fastapi import FastAPI, File
from src.ssd import build_ssd
from src.data_loaders import exp_cfg
import torch.nn as nn
from pdf2image import convert_from_bytes
from src.server.prediction_service import predict_from_images
from src.args import parse_test_args, get_gpus
import torch
import uvicorn
from fastapi.responses import StreamingResponse

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
                          "--trained_model","../trained_weights/ssd512GTDB_256_epoch15.pth"])

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
    return "Welcome to the deployed ScanSSD model, append '/docs' to this url to access the swagger web API"


@app.post("/predict/{dpi}")
def predict(dpi: int, file: bytes = File(...)):
    local_args = copy.deepcopy(args)
    images = convert_from_bytes(pdf_file=file, dpi=dpi)
    ret = predict_from_images(local_args, net, images)
    ret = pd.DataFrame(ret)
    out_file_as_str = StringIO()
    ret.to_csv(out_file_as_str, index=False, header=False, float_format='%.2f')
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

