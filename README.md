# 简介
图像分割算法实现：
- 一维最大熵算法
- 二维最大熵算法
- 结合边缘检测技术的改进一维最大熵算法
- 结合进化规划和边缘检测技术的一维最大熵算法
- otsu算法（直接使用OpenCV库内置函数）

# 安装与使用
prerequisite:
- python3.7
- Anaconda
- lib: numpy, OpenCV, matplotlib, [jupyter notebook]

```bash
$ cd img_seg
$ conda create --name img_seg --file requirements.txt
$ conda activate img_seg

$ python m1.py [img/<xxx>]
```
