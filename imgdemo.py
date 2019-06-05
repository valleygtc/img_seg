"""save img to img_demo/{img_name}-{origin/hist/m1/m1_border}
"""

import sys

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from utils import max_entropy_1d, calc_hist, calc_border_mask

img_names = ['img/lena_std.tif', 'img/cameraman.tif',
'img/lake.tif', 'img/airplane.jpg', 'img/elephant.jpg',
'img/Fig1038(a)(noisy_fingerprint).tif', 'img/house.tif', 'img/house2.tif', 'img/panther.jpg']

if __name__ == '__main__':
    for img_name in img_names:
        img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)

        hist_normal = calc_hist(img, normalized=True)

        border_mask = calc_border_mask(img, T_percent=0.2)
        masked_img_hist = cv.calcHist([img], [0], border_mask, [256], [0,256])
        masked_img_hist_normal = cv.normalize(masked_img_hist.ravel(), None, norm_type=cv.NORM_L1).ravel()

        m1_th = max_entropy_1d(hist_normal)
        m1_border_th = max_entropy_1d(masked_img_hist_normal)
        _, m1_th_img = cv.threshold(img, m1_th, 255, cv.THRESH_BINARY)
        _, m1_border_th_img = cv.threshold(img, m1_border_th, 255, cv.THRESH_BINARY)

        # 原图，灰度直方图，m1分割后的图，m1_border分割后的图。
        cv.imwrite('imgdemo/' + img_name.split('/')[-1].split('.')[0] + '-origin' + '.png', img)
        plt.hist(img.ravel(), 256, [0,256], density=True)
        plt.plot(m1_th, hist_normal[m1_th], 'ro')
        plt.plot(m1_border_th, hist_normal[m1_border_th], 'bo')
        plt.savefig('imgdemo/' + img_name.split('/')[-1].split('.')[0] + '-hist' + '.png', bbox_inches='tight')
        cv.imwrite('imgdemo/' + img_name.split('/')[-1].split('.')[0] + '-m1' + '.png', m1_th_img)
        cv.imwrite('imgdemo/' + img_name.split('/')[-1].split('.')[0] + '-m1_border' + '.png', m1_border_th_img)
        plt.close()
        print(f'{img_name} m1_th: {m1_th}, m1_border_th: {m1_border_th}')
