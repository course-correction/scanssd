from typing import List
from PIL.Image import Image
from src.data_loaders.gtdb_iterable import transform
import torch
import torch.nn.functional as F
import numpy as np
from src.utils.process_page import process_page


def predict_from_images(args, net, images: List[Image]):

    size = args.model_type
    mean = (246, 246, 246)
    mean = torch.ones((3, size, size), device='cpu') * mean[0]
    cv2_images = pil_to_cv2(images)

    all_page_windows = {str(page_id+1)+'_'+str(conf): [] for page_id in range(len(cv2_images))
                        for conf in args.conf}

    for img_id, img in enumerate(cv2_images):
        img_windows, pad_h, pad_w = transform(img, args.window, int(args.stride * args.window))
        for h in range(img_windows.shape[0]):
            for w in range(img_windows.shape[1]):
                # Resize the window to model input shape
                windows = img_windows[h, w].unsqueeze(0)
                windows = F.interpolate(windows, size=size,
                                        mode='area').squeeze()
                windows -= mean
                # Yield testing data_loaders
                # torch.cat((torch.tensor(9).unsqueeze(0), torch.tensor(9).clone().unsqueeze(0)))
                # yield h, w, pad_h, pad_w, tgt_win, img_id
                windows = torch.unsqueeze(windows, dim=0)
                predict_from_win(
                    args,
                    net,
                    windows,
                    torch.tensor(h).unsqueeze(0),
                    torch.tensor(w).unsqueeze(0),
                    torch.tensor(pad_h).unsqueeze(0),
                    torch.tensor(pad_w).unsqueeze(0),
                    torch.tensor(img_id).unsqueeze(0),
                    all_page_windows)

    return process_page(args, all_page_windows, 1, "", True)


def pil_to_cv2(pil_images: List[Image]) -> List:
    cv2_images = []
    for pil_image in pil_images:
        open_cv_image = np.array(pil_image.convert('RGB'))
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        cv2_images.append(open_cv_image)
    return cv2_images


def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())

def predict_from_win(args, net, windows, h_ids, w_ids, pad_hs, pad_ws, page_ids, all_page_windows):

    gpus = [str(gpu_id) for gpu_id in args.gpu]
    gpus = ','.join(gpus)
    devices = torch.device('cuda:' + gpus)

    images = windows.to(devices)
    y = net(images)  # forward pass
    detections = y.data.cpu().numpy()
    # Convert all metadata to numpy (these are NOT in GPU, so no gpu-cpu)
    h_ids = h_ids.numpy()
    w_ids = w_ids.numpy()
    pad_hs = pad_hs.numpy()
    pad_ws = pad_ws.numpy()
    base_offset = int(args.window * args.stride)
    # Threshold detections
    for idx, (h_id, w_id, pad_h, pad_w, page_id) in \
            enumerate(zip(h_ids, w_ids, pad_hs, pad_ws, page_ids)):

        #print(idx, h_id, w_id, pad_h, pad_w, page_id)
        img_id = page_id + 1
        y_l = h_id * base_offset
        x_l = w_id * base_offset

        pad_offset_x = int(pad_w/2)
        pad_offset_y = int(pad_h/2)

        for conf in args.conf:
            img_detections = detections[idx, 1, :, :]
            # Thredshold detections
            recognized_scores = img_detections[:,0]
            valid_mask = recognized_scores >= conf
            recognized_scores = recognized_scores[valid_mask]

            if len(recognized_scores):
                recognized_boxes = img_detections[:,1:] * args.window
                recognized_boxes = recognized_boxes[valid_mask]

                offset = np.array([x_l, y_l, x_l, y_l])
                pad_offset = np.array([pad_offset_x, pad_offset_y,
                                       pad_offset_x, pad_offset_y])
                recognized_boxes = recognized_boxes + offset - pad_offset

                all_page_windows[str(int(img_id))+'_'+str(conf)]\
                    .append([recognized_boxes, recognized_scores])