import sys
from timeit import timeit

import cv2 as cv

from utils import max_entropy_1d, max_entropy_1d_ep, calc_border_mask, calc_hist


def m1(img):
    hist_normal = calc_hist(img, normalized=True)
    threshold = max_entropy_1d(hist_normal)
    return threshold


def m1_border(img):
    border_mask = calc_border_mask(img, T_percent=0.2)
    masked_img_hist = cv.calcHist([img], [0], border_mask, [256], [0,256])
    masked_img_hist_normal = cv.normalize(masked_img_hist.ravel(), None, norm_type=cv.NORM_L1).ravel()
    threshold = max_entropy_1d(masked_img_hist_normal)
    return threshold


def m1_border_ep(img):
    border_mask = calc_border_mask(img, T_percent=0.2)
    masked_img_hist = cv.calcHist([img], [0], border_mask, [256], [0,256])
    masked_img_hist_normal = cv.normalize(masked_img_hist.ravel(), None, norm_type=cv.NORM_L1).ravel()
    threshold = max_entropy_1d_ep(masked_img_hist_normal)
    return threshold


img_name = 'img/lena_std.tif'
if len(sys.argv) > 1:
    img_name = sys.argv[1]
img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)

run_times = 1000
m1_time = timeit('m1(img)', number=run_times, globals=globals())
m1_border_time = timeit('m1_border(img)', number=run_times, globals=globals())
m1_border_ep_time = timeit('m1_border_ep(img)', number=run_times, globals=globals())

print(f'run {run_times} times')
print(f'm1_time: {m1_time}')
print(f'm1_border_time: {m1_border_time}')
print(f'm1_border_ep_time: {m1_border_ep_time}')
