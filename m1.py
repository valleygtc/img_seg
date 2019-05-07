"""使用一维最大熵算法计算图像阈值，完成图像切割。
"""

import sys

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from utils import max_entropy_1d, calc_hist


if __name__ == '__main__':
    img_name = 'img/lena_std.tif'
    if len(sys.argv) > 1:
        img_name = sys.argv[1]
    img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)

    hist_normal = calc_hist(img, normalized=True)
    threshold = max_entropy_1d(hist_normal)
    _, thr_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)

    # plot original gray image, hist, thr_img
    fig, ax_list = plt.subplots(1, 3)
    fig.suptitle('maximum entropy algorithem')
    ax_list[0].imshow(img, 'gray')

    ax_list[1].hist(img.ravel(), 256, [0,256], density=True)
    ax_list[1].plot(threshold, 0, 'ro')

    ax_list[2].imshow(thr_img, 'gray')

    print(f'图像 {img_name} 的阈值为: {threshold}')
    plt.show()
