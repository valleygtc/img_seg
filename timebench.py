import sys
from timeit import timeit

import cv2 as cv

from utils import max_entropy_1d, max_entropy_1d_ep, calc_border_mask


img_name = 'img/lena_std.tif'
if len(sys.argv) > 1:
    img_name = sys.argv[1]
img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)

border_mask = calc_border_mask(img, T_percent=0.2)
masked_img_hist = cv.calcHist([img], [0], border_mask, [256], [0,256])
masked_img_hist_normal = cv.normalize(masked_img_hist.ravel(), None, norm_type=cv.NORM_L1).ravel()

run_times = 1000
normal_time = timeit('max_entropy_1d(masked_img_hist_normal)', number=run_times, globals=globals())
ep_time = timeit('max_entropy_1d_ep(masked_img_hist_normal)', number=run_times, globals=globals())

print(f'run {run_times} times')
print(f'normal_time: {normal_time}')
print(f'ep_time: {ep_time}')
