import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


def salt_noise(img):
    mask_arr = np.random.rand(img.shape[0], img.shape[1])
    img[:, :, 0][mask_arr <= 0.4] = 255
    img[:, :, 1][mask_arr <= 0.4] = 255
    img[:, :, 2][mask_arr <= 0.4] = 255
    plt.imshow(img)
    plt.show()
    plt.savefig('Test_Figures/Salt_Noise_40.png', dpi=300, bbox_inches='tight')


def band_noise(img):
    gt_file = 'eval/chem_gt_small/US20050043351A1.csv'
    gt_data = np.genfromtxt(gt_file, delimiter=',')
    gt_data = gt_data[gt_data[:, 0] == 12][:, 1:]
    gt_data = gt_data.astype('int32')

    width_min = 50
    width_max = 100
    noise_prob = 0.70
    width_range = width_max - width_min
    width = int((random.random() * width_range)) + width_min

    for box in gt_data:
        if random.random() <= noise_prob:
            box_height = box[3] - box[1]
            start_band = int((random.random() * box_height) + box[1])
            img[start_band:start_band + width, :, :] = 255

    for box in gt_data:
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 5)

    plt.imshow(img)
    plt.show()
    plt.savefig('Test_Figures/Band_Noise_40.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    img_path = '/home/abhisek/Desktop/MathSeer-extraction-pipeline/src/ScanSSD' \
                 '/chem_data/images/US20050043351A1/13.png'
    image = cv2.imread(img_path, 1)
    # salt_noise(image)
    band_noise(image)
