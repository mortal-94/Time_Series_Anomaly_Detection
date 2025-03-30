import numpy as np
from pyextremes import EVA


def get_threshold(errors, initial_threshold, confidence_level=0.99):

    """
    用POT方法计算阈值
    :param errors: 误差数据
    :param initial_threshold: 初始化阈值
    :param confidence_level: 置信水平，默认值为0.99，表示保证99%的异常误差大于该阈值
    :return: 阈值
    """
    # 创建EVA对象并拟合POT模型
    model = EVA(data=errors)
    model.get_extremes(method="POT", threshold=initial_threshold)
    model.fit_model()

    # 获取POT方法的阈值
    threshold = model.get_threshold(confidence_level)
    return threshold