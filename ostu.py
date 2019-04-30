import sys

import cv2 as cv
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


img_name = 'img/lena_std.tif'
if len(sys.argv) > 1:
    img_name = sys.argv[1]

img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
# Otsu's thresholding
threshold, thr_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
threshold = int(threshold)
# 灰度直方图
hist = cv.calcHist([img], [0], None, [256], [0,256])

# plot gray image, hist, thr_img
fig, ax_list = plt.subplots(1, 3)
fig.suptitle('ostu algorithem')
ax_list[0].imshow(img, 'gray')

ax_list[1].hist(img.ravel(), 256, [0,256], density=True)
ax_list[1].plot(threshold, 0, 'ro')

ax_list[2].imshow(thr_img, 'gray')

print(f'threshod calculated by ostu: {threshold}')
plt.show()
