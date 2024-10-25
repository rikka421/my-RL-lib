import numpy as np
import matplotlib.pyplot as plt

class MDP:
    # 此处考虑离散的状态和动作空间, 于是将S, A, P, R都采用了列表或字典形式;
    # 之后对于连续空间可能采用神经网络之类的建模.
    def __init__(self, states_num, actions_num, transition_probs, rewards, start_state, terminal_states):
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
        reward = self.get_reward(state, action)
        next_state = self.get_next_state(state, action)
        done = (state in self.termination_states)
        return next_state, reward, done
