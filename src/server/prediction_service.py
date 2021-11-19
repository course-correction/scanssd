from typing import List
from PIL.Image import Image
import numpy

def predict(net, images: List[Image]):
    cv2_images = pil_to_cv2(images)



def pil_to_cv2(pil_images: List[Image]) -> List:
    cv2_images = []
    for pil_image in pil_images:
        open_cv_image = numpy.array(pil_image.convert('RGB'))
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        cv2_images.append(open_cv_image)
    return cv2_images


