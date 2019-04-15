import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


# 밴딧의 손잡이 목록을 작성한다.
# 현재 손잡이 4(인덱스 3)가 가장 자주 양의 보상을 제공하도록 설정되어 있다.
bandit_arms = [0.2, 0, -0.2, -2]
num_arms = len(bandit_arms)


def pullBandit(bandit):
    # 랜덤한 값을 구한다.
    result = np.random.randn(1)
    if result > bandit:
        # 양의 보상을 반환 한다.
        return 1
    else:
        # 음의 보상을 반환 한다.
        return -1