import sys

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def max_entropy(hist_normal):
    """
    Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method
    Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for Gray-Level Picture Thresholding Using the Entropy
    of the Histogram", Graphical Models and Image Processing, 29(3): 273-285
    参考：https://github.com/zenr/ippy/blob/master/segmentation/max_entropy.py
    Params:
        hist_normal [np.array]: 归一化后的图像灰度直方图。
    Return:
        threshold [int]: threshold calculated by 一维最大熵算法
    """

    # calculate normalized CDF (cumulative density function)
    cdf_normal = hist_normal.cumsum()

    valid_range = np.nonzero(hist_normal)[0]
    s_range = hist_normal[hist_normal != 0]
    H_s_cum = -np.cumsum(s_range * np.log(s_range))

    H_n = H_s_cum[-1]

    max_ent, threshold = 0, 0
    for i in range(len(H_s_cum) - 1): # 忽略最后一个非零点，防止P_s为1导致(1 - P_s)为0
        s = valid_range[i]
        P_s = cdf_normal[s]
        H_s = H_s_cum[i]
        total_ent = np.log(P_s * (1 - P_s)) + H_s/P_s + (H_n - H_s)/(1 - P_s)

        # find max
        if total_ent > max_ent:
            max_ent, threshold = total_ent, s

    return threshold

img_name = 'img/lena_std.tif'
if len(sys.argv) > 1:
    img_name = sys.argv[1]
img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)

lap_img = abs(cv.Laplacian(img, -1))
lap_hist = cv.calcHist([lap_img], [0], None, [256], [0,256])

T = np.max(lap_img) * 0.2 # 阈值取拉普拉斯图像中最大值的xx%
_, mask = cv.threshold(lap_img, T, 255, cv.THRESH_BINARY)
masked_img = cv.bitwise_and(img, img, mask=mask)
masked_img_hist = cv.calcHist([img], [0], mask, [256], [0,256])
masked_img_hist_normal = cv.normalize(masked_img_hist.ravel(), None, norm_type=cv.NORM_L1).ravel()

threshold = max_entropy(masked_img_hist_normal)
_, thr_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)

# plot gray image, hist, lap_img, mask, masked_img, masked_img_hist, thr_img
fig, ax_list = plt.subplots(4, 2)
ax_list[0][0].imshow(img, 'gray')
ax_list[0][0].set_title('original img')
ax_list[0][1].hist(img.ravel(), 256, [0,256], density=True)
ax_list[0][1].set_title('origin img hist')
ax_list[0][1].plot(threshold, 0, 'ro')

ax_list[1][0].imshow(lap_img, 'gray')
ax_list[1][0].set_title('lap_img')
ax_list[1][1].imshow(mask, 'gray')
ax_list[1][1].set_title('mask')

ax_list[2][0].imshow(masked_img, 'gray')
ax_list[2][0].set_title('masked_img')
ax_list[2][1].plot(masked_img_hist)
ax_list[2][1].plot(threshold, 0, 'ro')
ax_list[2][1].set_title('masked_img_hist')

ax_list[3][0].imshow(thr_img, 'gray')
ax_list[3][0].set_title('thr_img')

print(f'经过边缘改良后的最大熵计算阈值为：{threshold}')
plt.show()
