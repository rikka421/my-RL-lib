import numpy as np


class Policy:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.num_states = len(states)
        self.num_actions = len(actions)
        # 初始化策略为均匀随机选择  # Dict: {state: {action: probability}}
        self.policy = {state: np.ones(self.num_actions) / self.num_actions for state in states}
        # self.policy = {state: {action: np.random.random() for action in actions} for state in states}

    def choose_action(self, state):
        # 返回当前状态下的动作
        return np.random.choice(self.actions, p=self.policy[state])

    def update_policy(self, state, probabilities):
        # 更新某个状态的策略
        self.policy[state] = probabilities

    def print(self):
        print(self.policy)


if __name__ == "__main__":
    # 示例用法
    states = ['A', 'B']
    actions = ['move', 'stay']

    policy = Policy(states, actions)
    policy.print()

    current_state = 'A'
    for _ in range(5):
        action = policy.choose_action(current_state)
        print(f"{_} state {current_state}, action: {action}")

    policy.update_policy(current_state, [0.1, 0.9])
    for _ in range(5):
        action = policy.choose_action(current_state)
        print(f"{_} state {current_state}, action: {action}")