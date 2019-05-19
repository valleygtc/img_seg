import numpy as np
import cv2 as cv
import ep


def max_entropy_1d(hist_normal):
    """
    Implement Kapur-Sahoo-Wong (1d Maximum Entropy) thresholding method.
    paper: J. N. KAPUR, P. K. SAHOO, A. K. C. WONG.
    A New Method for Gray-Level Picture Thresholding Using the Entropy of the Histogram[J].
    COMPUTER VISION, GRAPHICS, AND IMAGE PROCESSING, 1985, 29:273-285.

    Params:
        hist_normal [1d np.array]: 归一化后的图像灰度直方图。
    Return:
        threshold [int]
    """

    # calculate normalized CDF (cumulative density function)
    cdf_normal = hist_normal.cumsum()

    valid_idxs = np.nonzero(hist_normal)[0]
    valid_p_i = hist_normal[valid_idxs]
    H_s_cum = -np.cumsum(valid_p_i * np.log(valid_p_i))

    H_n = H_s_cum[-1]

    max_ent, threshold = 0, 0
    for i in range(len(valid_idxs) - 1): # 忽略最后一个非零点，防止P_s为1导致(1 - P_s)为0
        s = valid_idxs[i]
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
    # lap_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    # lap_img = abs(cv.filter2D(img, -1, lap_kernel))
    lap_hist = cv.calcHist([lap_img], [0], None, [256], [0,256])

    T = np.max(lap_img) * T_percent # 阈值取拉普拉斯图像中最大值的xx%
    _, border_mask = cv.threshold(lap_img, T, 255, cv.THRESH_BINARY)
    return border_mask


def max_entropy_1d_ep(hist_normal, population_size=10, iter_num=5, competition_q=9):
    """结合进化规划的一维最大熵算法：根据图像灰度直方图计算阈值。
    Params:
        hist_normal [np.array]: 归一化后的图像灰度直方图。
        population_size [int]: 一个种群内个体数目
        iter_num [int]:
        competition_q [int]: q-竞争法选择下一代。
    Return:
        threshold [int]:
    """

    p_i = hist_normal
    P_s = p_i.cumsum()

    valid_i = np.nonzero(p_i)[0]
    valid_p_i = p_i[valid_i]
    valid_P_s = P_s[valid_i]
    valid_H_s = -np.cumsum(valid_p_i * np.log(valid_p_i))

    H_n = valid_H_s[-1]

    def calc_ents(valid_i_idxs):
        P_s = valid_P_s[valid_i_idxs]
        H_s = valid_H_s[valid_i_idxs]
        ents = np.log(P_s * (1 - P_s)) + H_s/P_s + (H_n - H_s)/(1 - P_s)
        return ents

    # 将valid_i_idx作为搜索空间来使用进化规划搜索最优解。
    max_ent, threshold = 0, 0
    valid_i_len = len(valid_i) - 1# 忽略最后一个非零点，防止P_s为1导致(1 - P_s)为0
    # 初始种群
    valid_i_idxs = ep.get_init_population(np.arange(0, valid_i_len), population_size)
    ents = calc_ents(valid_i_idxs)
    for i in range(iter_num): # 迭代终止条件为：指定的迭代次数
        # 变异
        new_valid_i_idxs =  ep.mutate(
            valid_i_idxs, 
            fitnesses=cv.normalize(ents, None, 0, valid_i_len - 1, cv.NORM_MINMAX).ravel(),
            beta=1,
            gamma=0) % valid_i_len
        new_ents = calc_ents(new_valid_i_idxs)
        # 选择：使用q-竞争法选择出I个个体组成的种群。
        total_valid_i_idxs = np.concatenate((valid_i_idxs, new_valid_i_idxs))
        total_ents = np.concatenate((ents, new_ents))
        survive_idxs = ep.select(total_ents, population_size, q=9)
        survive_population = total_valid_i_idxs[survive_idxs]
        survive_ents = total_ents[survive_idxs]
        valid_i_idxs = survive_population
        ents = survive_ents

    threshold = valid_i[valid_i_idxs[-1]]
    return threshold


def max_entropy_2d(hist_2d):
    """
    Implements AHMED S. ABLJTALEB* (2d Maximum Entropy) thresholding method.
    paper: AHMED S. ABLJTALEB* (1989) "Automatic Thresholding of Gray-Level Pictures Using Two-Dimensional Entropy"
    Params:
        hist_2d [np.array]: 归一化后的二维直方图，i, j为两个描绘子。一般维i：像素灰度值，j：像素x*x邻域平均灰度值。
    Return:
        threshold [int]: threshold calculated by 2维最大熵算法
    """

    valid_idxs = np.nonzero(hist_2d)

    valid_p_ij = hist_2d[valid_idxs]
    p_ij_cum = np.cumsum(valid_p_ij)
    H_s_cum = -np.cumsum(valid_p_ij * np.log(valid_p_ij))
    H_mm = H_s_cum[-1]
    max_ent, threshold = 0, 0
    for i in range(len(valid_p_ij) - 1):  # 忽略最后一个非零点，防止P_st为1导致(1 - P_st)为0
        s, t = valid_idxs[0][i], valid_idxs[1][i]
        P_st = p_ij_cum[i] #?为什么论文中这里是负的？
        H_st = H_s_cum[i]
        total_ent = np.log(P_st * (1 - P_st)) + H_st/P_st + (H_mm - H_st)/(1 - P_st)
        # find max
        if total_ent > max_ent:
            max_ent, threshold = total_ent, s

    return threshold
