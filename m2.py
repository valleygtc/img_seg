"""使用二维最大熵算法计算图像阈值，完成图像切割。
"""

import sys

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from utils import max_entropy_2d


if __name__ == '__main__':
    img_name = 'img/lena_std.tif'
    if len(sys.argv) > 1:
        img_name = sys.argv[1]
    img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    hist_2d, xbins, ybins = np.histogram2d(img.ravel(), cv.blur(img, (3, 3)).ravel(), bins=256, range=((0, 256), (0, 256)), density=True)
    threshold = max_entropy_2d(hist_2d)

    _, thr_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)

    # plot gray image, hist, thr_img
    fig, ax_list = plt.subplots(1, 4)
    fig.suptitle('2d maximum entropy algorithem')
    ax_list[0].imshow(img, 'gray')
    ax_list[0].set_title('original img')

    ax_list[1].imshow(cv.blur(img, (3, 3)), 'gray')
    ax_list[1].set_title('blurred img')

    ax_list[2].hist2d(img.ravel(), cv.blur(img, (3, 3)).ravel(), bins=256, range=((0, 256), (0, 256)), normed=True)

    ax_list[3].imshow(thr_img, 'gray')

    print(f'图像 {img_name} 的阈值为: {threshold}')
    plt.show()
