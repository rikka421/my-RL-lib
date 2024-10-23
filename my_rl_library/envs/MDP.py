import numpy as np
import matplotlib.pyplot as plt

class MDP:
    # 此处考虑离散的状态和动作空间, 于是将S, A, P, R都采用了列表或字典形式;
    # 之后对于连续空间可能采用神经网络之类的建模.
    def __init__(self, states, actions, transition_probs, rewards):
        # 此处我们没有建模gamma, 而是将其放置在策略中进行考虑
        self.states = states
        self.start_state = self.states[0]
        self.actions = actions
        self.transition_probs = transition_probs  # Dict: {state: {action: {next_state: probability}}}
        self.rewards = rewards  # Dict: {(state, action): reward}

    def get_next_states_probs(self, state, action):
        # 显式建模了get_states, 方便后续类继承
        return self.transition_probs.get(state, {}).get(action, {})

    def get_reward(self, state, action):
        # 显式建模了get_reward, 方便后续类继承
        return self.rewards.get((state, action), 0)

    def step(self, state, action):
        next_states_probs = self.get_next_states_probs(state, action)
        next_states = list(next_states_probs.keys())
        probs = list(next_states_probs.values())
        reward = self.get_reward(state, action)
        if sum(probs) == 0:
            return state, reward, True
        next_state = np.random.choice(next_states, p=probs)
        return next_state, reward, False

    def print(self):
        print(self.states, self.actions)
        print(self.transition_probs)
        print(self.rewards)


if __name__ == "__main__":
    # 示例用法
    states = ['A', 'B']
    actions = ['move', 'stay']
    transition_probs = {
        'A': {'move': {'B': 1.0}, 'stay': {'A': 1.0}},
        'B': {'move': {'A': 1.0}, 'stay': {'B': 1.0}}
    }
    rewards = {('A', 'move'): 1, ('B', 'move'): 0}

    mdp = MDP(states, actions, transition_probs, rewards)
    print("some thing change")
