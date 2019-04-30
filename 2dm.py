import sys

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

def max_entropy_2d(hist_2d):
    """
    Implements AHMED S. ABLJTALEB* (2d Maximum Entropy) thresholding method
    AHMED S. ABLJTALEB* (1989) "Automatic Thresholding of Gray-Level Pictures Using Two-Dimensional Entropy"
    Params:
        hist_2d [np.array]: 归一化后的二维直方图，i：像素灰度值，j：像素3*3邻域平均灰度值。
    Return:
        threshold [int]: threshold calculated by 2维最大熵算法
    """

    nonzero_indices = np.nonzero(hist_2d)
    i_start, i_end = nonzero_indices[0][0], nonzero_indices[0][-1]
    j_start, j_end = nonzero_indices[1][0], nonzero_indices[1][-1]

    total_range = hist_2d[hist_2d != 0]
    H_mm = -np.sum(total_range * np.log(total_range))
    max_ent, threshold = 0, 0
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            st_range = hist_2d[:i+1,:j+1]
            st_range = st_range[st_range != 0]
            P_st = np.sum(st_range) #?为什么论文中这里是负的？
            H_st = -np.sum(st_range * np.log(st_range))
            total_ent = np.log(P_st * (1 - P_st)) + H_st/P_st + (H_mm - H_st)/(1 - P_st)
            # find max
            if total_ent > max_ent:
                max_ent, threshold = total_ent, i

    return threshold
    


img_name = 'img/lena_std.tif'
if len(sys.argv) > 1:
    img_name = sys.argv[1]
img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
hist_2d, xbins, ybins = np.histogram2d(img.ravel(), cv.blur(img, (3, 3)).ravel(), bins=256, range=((0, 256), (0, 256)), density=True)
threshold = max_entropy_2d(hist_2d)

_, thr_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)

# plot gray image, hist, thr_img
fig, ax_list = plt.subplots(1, 4)
fig.suptitle('maximum entropy algorithem')
ax_list[0].imshow(img, 'gray')
ax_list[1].imshow(cv.blur(img, (3, 3)), 'gray')

ax_list[2].hist2d(img.ravel(), cv.blur(img, (3, 3)).ravel(), bins=256, range=((0, 256), (0, 256)), normed=True)

ax_list[3].imshow(thr_img, 'gray')

print(f'threshod calculated by  二维最大熵: {threshold}')
plt.show()
