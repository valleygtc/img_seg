import numpy as np
import cv2 as cv


def max_entropy_1d(hist_normal):
    """
    Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method
    Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for Gray-Level Picture Thresholding Using the Entropy
    of the Histogram", Graphical Models and Image Processing, 29(3): 273-285
    Params:
        hist_normal [1d np.array]: 归一化后的图像灰度直方图。
    Return:
        threshold [int]:
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


def calc_hist(img, normalized=False):
    """caculate image gray level histogram. gray level: 0-255.
    Params:
        img [np.array]
        normalized [bool]
    Return:
        hist [np.array]
    """
    hist = cv.calcHist([img], [0], None, [256], [0,256])
    if normalized:
        hist_normal = cv.normalize(hist.ravel(), None, norm_type=cv.NORM_L1).ravel()
        return hist_normal
    else:
        hist


def calc_border_mask(img, T_percent):
    """计算得到边缘附近像素点位置的掩模。
    1. 使用拉普拉斯算子计算得到img的边缘图像
    2. 选取边缘图像最大值的T_percent作为阈值对边缘图像做阈值处理，得到一幅二值图像。该二值图像即为结果（亮点即可认为是边缘附近的像素点）
    """
    lap_img = abs(cv.Laplacian(img, -1))
    lap_hist = cv.calcHist([lap_img], [0], None, [256], [0,256])

    T = np.max(lap_img) * T_percent # 阈值取拉普拉斯图像中最大值的xx%
    _, border_mask = cv.threshold(lap_img, T, 255, cv.THRESH_BINARY)
    return border_mask


def max_entropy_1d_ep(hist_normal, group_num=10, iter_num=5, competition_q=9):
    """结合进化规划的一维最大熵算法：根据图像灰度直方图计算阈值。
    Params:
        hist_normal [np.array]: 归一化后的图像灰度直方图。
        group_num [int]: 一个种群内个体数目
        iter_num [int]:
        competition_q [int]: q-竞争法选择下一代。
    Return:
        threshold [int]:
    """

    # calculate normalized CDF (cumulative density function)
    cdf_normal = hist_normal.cumsum()

    valid_range = np.nonzero(hist_normal)[0]
    s_range = hist_normal[valid_range]
    H_s_cum = -np.cumsum(s_range * np.log(s_range))

    H_n = H_s_cum[-1]

    max_ent, threshold = 0, 0
    H_s_cum_len = len(H_s_cum)
    # 初始种群
    group_idxs = np.random.randint(0, H_s_cum_len - 1, size=group_num)
    group_s = valid_range[group_idxs]
    group_P_s = cdf_normal[group_s]
    group_H_s = H_s_cum[group_idxs]
    group_ents = np.log(group_P_s * (1 - group_P_s)) + group_H_s/group_P_s + (H_n - group_H_s)/(1 - group_P_s)
    for i in range(iter_num): # 迭代终止条件为：迭代次数5
        # 变异
        new_group_idxs =  (np.round(group_idxs + np.random.sample(group_num) * np.sqrt(np.std(group_idxs))) % H_s_cum_len).astype(int)
        new_group_s = valid_range[new_group_idxs]
        new_group_P_s = cdf_normal[new_group_s]
        new_group_H_s = H_s_cum[new_group_idxs]
        new_group_ents = np.log(group_P_s * (1 - group_P_s)) + group_H_s/group_P_s + (H_n - group_H_s)/(1 - group_P_s)
        # 选择：使用q-竞争法选择出I个个体组成的种群。
        total_group_idxs = np.concatenate((group_idxs, new_group_idxs))
        total_group_ents = np.concatenate((group_ents, new_group_ents))
        q_test_group_ents = np.random.choice(total_group_ents, competition_q, replace=False)
        def test(ent, q_test_group_ents):
            score = 0
            for test_ent in q_test_group_ents:
                if ent > test_ent:
                    score += 1
            return score

        total_group_score = [test(ent, q_test_group_ents) for ent in total_group_ents]
        survive_idxs = np.argsort(total_group_score)[-group_num:]
        survive_group_idxs = total_group_idxs[survive_idxs]
        survive_group_ents = total_group_ents[survive_idxs]
        group_idxs = survive_group_idxs
        group_ents = survive_group_ents

    threshold = valid_range[group_idxs[-1]]
    return threshold


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
