"""进化规划
"""

import numpy as np
import cv2 as cv

"""# TODO
population:
[individual, individual, ...]

individual:
[x, target, fitness]
"""


def get_init_population(search_space, size):
    """获取初始种群，种群的个体数为<size>。
    Params:
        search_space [np.array]: 1d
        size [int]
    Return:
        population [np.array]: 1d
    """
    return np.random.choice(search_space, size=size)


def calc_fitnesses(population, fitness_func):
    return fitness_func(population)


def mutate(population, fitnesses, beta=1, gamma=0):
    """使用高斯变异（Gauss mutation）算子
    Params:
        population [np.array]: 1d
    Return:
        new_population [np.array]: 1d
    """
    sigma = np.sqrt(fitnesses * beta + gamma)
    gauss_std_distribution = np.random.normal(0, 1, size=len(population))
    return (population + sigma * gauss_std_distribution).astype(population.dtype)


def select(total_fitnesses, select_size, q=None):
    """使用q-竞争法选择
    Params:
        total_fitnesses [np.array]: 1d, 包含父种群和变异出的子种群。
        select_size [int]:
        q [int]:
    Return:
        选择出的个体的idxs
    """
    if q is None:
        q = int(select_size * 0.9)

    def score(fitness, q_test_fitnesses):
        score = 0
        for f in q_test_fitnesses:
            if fitness > f:
                score += 1
        return score
    q_test_fitnesses = np.random.choice(total_fitnesses, q, replace=False)
    scores = [score(fitness, q_test_fitnesses) for fitness in total_fitnesses]
    return np.argsort(scores)[-select_size:]