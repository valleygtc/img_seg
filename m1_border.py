"""由使用边缘信息改进后的一维最大熵算法计算图像阈值，完成图像切割。
"""

import sys

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from utils import max_entropy_1d, calc_border_mask


if __name__ == '__main__':
    img_name = 'img/lena_std.tif'
    if len(sys.argv) > 1:
        img_name = sys.argv[1]
    img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)

    border_mask = calc_border_mask(img, T_percent=0.2)
    masked_img_hist = cv.calcHist([img], [0], border_mask, [256], [0,256])
    masked_img_hist_normal = cv.normalize(masked_img_hist.ravel(), None, norm_type=cv.NORM_L1).ravel()

    threshold = max_entropy_1d(masked_img_hist_normal)
    _, thr_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)

    # plot original gray image, hist, thr_img
    plt.subplot(1, 3, 1)
    plt.suptitle('maximum entropy algorithem(use border info optimized)')
    plt.imshow(img, 'gray')
    plt.title('original img')

    plt.subplot(1, 3, 2)
    plt.hist(img.ravel(), 256, [0,256], density=True)
    plt.title('origin img hist')
    plt.plot(threshold, 0, 'ro')

    plt.subplot(1, 3, 3)
    plt.imshow(thr_img, 'gray')
    plt.title('thr_img')

    print(f'图像 {img_name} 的阈值为：{threshold}')
    plt.show()
