"""对比枚举法和进化规划用于一维最大熵算法的时间、阈值
"""

import sys
from timeit import timeit

import cv2 as cv

from utils import max_entropy_1d, max_entropy_1d_ep, calc_border_mask, calc_hist


img_name = 'img/lena_std.tif'
if len(sys.argv) > 2:
    img_name = sys.argv[2]
img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)

border_mask = calc_border_mask(img, T_percent=0.2)
masked_img_hist = cv.calcHist([img], [0], border_mask, [256], [0,256])
masked_img_hist_normal = cv.normalize(masked_img_hist.ravel(), None, norm_type=cv.NORM_L1).ravel()

if sys.argv[1] == '--time':
    run_times = 1000
    enum_time = timeit('max_entropy_1d(masked_img_hist_normal)', number=run_times, globals=globals())
    ep_time = timeit('max_entropy_1d_ep(masked_img_hist_normal)', number=run_times, globals=globals())

    print(f'run {run_times} times')
    print(f'enum_time: {enum_time}')
    print(f'ep_time: {ep_time}')
elif sys.argv[1] == '--threshold':
    run_times = 1000
    ep_results = {}
    for i in range(run_times):
        t = max_entropy_1d_ep(masked_img_hist_normal, population_size=10, iter_num=10, competition_q=9)
        if t in ep_results:
            ep_results[t] += 1
        else:
            ep_results[t] = 1
    enum_result = max_entropy_1d(masked_img_hist_normal)
    print(f'run {run_times} times')
    print(f'enum_result: {enum_result}')
    print(f'ep_results: {ep_results}')
    ep_average = 0
    for t, num in ep_results.items():
        ep_average += t * num/run_times
    print(f'ep_average: {ep_average}')
