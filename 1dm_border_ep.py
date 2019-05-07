import sys

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def max_entropy(hist_normal):
    """结合进化规划的一维最大熵算法：根据图像灰度直方图计算阈值。
    Params:
        hist_normal [np.array]: 归一化后的图像灰度直方图。
    Return:
        threshold [int]: threshold calculated by 一维最大熵算法
    """

    # calculate normalized CDF (cumulative density function)
    cdf_normal = hist_normal.cumsum()

    valid_range = np.nonzero(hist_normal)[0]
    s_range = hist_normal[valid_range]
    H_s_cum = -np.cumsum(s_range * np.log(s_range))

    H_n = H_s_cum[-1]

    max_ent, threshold = 0, 0
    H_s_cum_len = len(H_s_cum)
    group_idxs = np.random.randint(0, H_s_cum_len - 1, size=10) # 种群数为10

    for i in range(30): # 迭代终止条件为：迭代次数30
        group_s = valid_range[group_idxs]
        group_P_s = cdf_normal[group_s]
        group_H_s = H_s_cum[group_idxs]
        group_ents = np.log(group_P_s * (1 - group_P_s)) + group_H_s/group_P_s + (H_n - group_H_s)/(1 - group_P_s)
        # 变异
        new_group_idxs =  (np.round(group_idxs + np.random.sample(10) * np.sqrt(np.std(group_idxs))) % H_s_cum_len).astype(int)
        new_group_s = valid_range[new_group_idxs]
        new_group_P_s = cdf_normal[new_group_s]
        new_group_H_s = H_s_cum[new_group_idxs]
        new_group_ents = np.log(group_P_s * (1 - group_P_s)) + group_H_s/group_P_s + (H_n - group_H_s)/(1 - group_P_s)
        # 选择：使用q-竞争法选择出I个个体组成的种群。
        total_group_idxs = np.concatenate((group_idxs, new_group_idxs))
        total_group_ents = np.concatenate((group_ents, new_group_ents))
        q_test_group_ents = np.random.choice(total_group_ents, 9, replace=False)
        def test(ent, q_test_group_ents):
            score = 0
            for test_ent in q_test_group_ents:
                if ent > test_ent:
                    score += 1
            return score

        total_group_score = [test(ent, q_test_group_ents) for ent in total_group_ents]
        survive_group_idxs = total_group_idxs[np.argsort(total_group_score)[-10:]]
        group_idxs = survive_group_idxs

    threshold = valid_range[group_idxs[-1]]
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
