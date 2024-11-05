import numpy as np

class ValueEstimator:
    def __init__(self):
        # 价值估算器. 白盒环境下, 应当传入: P, pi, R.
        pass

    def update(self):
        raise NotImplementedError

    def return_value(self):
        raise NotImplementedError

class NativeValueEstimator(ValueEstimator):
    def compute(self, P, pi, R, gamma, states_num, actions_num):
        """ 利用贝尔曼方程的矩阵形式计算解析解, states_num是MRP的状态数
            其中P = R^ s*s, pi = R^ s*a"""
        P = np.array(P)
        pi = np.array(pi)
        R = np.array(R)

        E_R = np.sum(pi * R, axis=1)    # sum (s, a)  = (s, 1)

        pi_P = gamma * np.sum(P * pi[:, :, np.newaxis], axis=1)  # sum (s, a, s)  = (s, s)

        value = np.linalg.inv(np.eye(states_num, states_num) - pi_P) @ E_R   # (s, s) @ (s, 1) = (s, 1)

        return value

class MCEstimator(ValueEstimator):
    def __init__(self):
        pass

class TDEstimator(ValueEstimator):
    def __init__(self):
        pass

import numpy as np
from my_rl_library.envs.MDPEnv.RandomWalkMDP import RandomWalkMDP
if __name__ == '__main__':

    estimator = NativeValueEstimator()
    mdp = RandomWalkMDP(5)


    pi = np.full((mdp.states_num, mdp.actions_num), 1/mdp.actions_num)

    result = estimator.compute(mdp.transition_probs,
                               pi, mdp.rewards, 1,
                               mdp.states_num,
                               mdp.actions_num)
    # result = np.sum(A, axis=1)
    print(result)





