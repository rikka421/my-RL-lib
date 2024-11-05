import numpy as np
from my_rl_library.envs.Env import MyEnv

class MDP(MyEnv):
    # 此处考虑离散的状态和动作空间, 于是将S, A, P, R都采用了列表或字典形式;
    # 格子世界的写法.
    # 之后对于连续空间可能采用神经网络之类的建模.
    def __init__(self, states_num, actions_num, transition_probs, rewards, start_state, terminal_states):
        super(MDP, self).__init__()
        # 使用矩阵来建模转移过程.
        # P: P(S_i 到 S_j) = P_ij
        # R: 可能需要两个参数, 也可能需要一个参数. 子类中各不相同.
        self.states_num = states_num
        self.actions_num = actions_num
        self.transition_probs = transition_probs
        self.rewards = rewards

        self.start_state = start_state
        self.termination_states = terminal_states

    def get_reward(self, state, action):
        return self.rewards[state][action]

    def get_next_state(self, state, action):
        return np.random.choice(
            list(range(self.states_num)),
            p=self.transition_probs[state, action])

    def reset(self):
        return self.start_state

    def step(self, state, action):
        """
        此处函数和一般的Gym Env函数很不相同, 接收了一个state作为参数.
        因此比起真正的环境, 更像是一个建模出的仿真环境. 但是真正建模环境的时候一般使用的是神经网络,
        所以定位还挺尴尬的.
        :param state:
        :param action:
        :return:
        """
        reward = self.get_reward(state, action)
        next_state = self.get_next_state(state, action)
        done = (state in self.termination_states)
        return next_state, reward, done
