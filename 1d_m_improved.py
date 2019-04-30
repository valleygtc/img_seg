import sys

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def max_entropy(hist):
    """
    Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method
    Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for Gray-Level Picture Thresholding Using the Entropy
    of the Histogram", Graphical Models and Image Processing, 29(3): 273-285
    参考：https://github.com/zenr/ippy/blob/master/segmentation/max_entropy.py
    Params:
        hist [np.array]: Sequence representing the histogram of the image
    Return:
        threshold [int]: threshold calculated by 最大熵算法
    """

    # calculate CDF (cumulative density function)
    cdf = hist.cumsum()
    cdf_normal = cdf / cdf[-1]
    hist_normal = hist / cdf[-1]

    valid_range = np.nonzero(hist_normal)[0]
    start, end = valid_range[0], valid_range[-1]

    max_ent, threshold = 0, 0
    total_range = hist_normal[hist_normal != 0]
    H_n = -np.sum(total_range * np.log(total_range))
    for s in range(start, end):
        P_s = cdf_normal[s]
        s_range = hist_normal[:s+1]
        s_range = s_range[s_range != 0]
        H_s = -np.sum(s_range * np.log(s_range))
        total_ent = np.log(P_s * (1 - P_s)) + H_s/P_s + (H_n - H_s)/(1 - P_s)

        # find max
        if total_ent > max_ent:
            max_ent, threshold = total_ent, s

    return threshold

img_name = 'img/lena_std.tif'
if len(sys.argv) > 1:
    img_name = sys.argv[1]
img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)

hist = cv.calcHist([img], [0], None, [256], [0,256])
threshold = max_entropy(hist.ravel())
_, thr_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)

# plot gray image, hist, thr_img
fig, ax_list = plt.subplots(1, 3)
fig.suptitle('maximum entropy algorithem')
ax_list[0].imshow(img, 'gray')

ax_list[1].hist(img.ravel(), 256, [0,256], density=True)
ax_list[1].plot(threshold, 0, 'ro')

ax_list[2].imshow(thr_img, 'gray')

print(f'threshod calculated by 最大熵: {threshold}')
plt.show()
