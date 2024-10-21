import numpy as np


class Policy:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.num_states = len(states)
        self.num_actions = len(actions)
        # 初始化策略为均匀随机选择  # Dict: {state: {action: probability}}
        self.policy = {state: {action: 1 / self.num_actions for action in actions} for state in states}

    def get_pro(self, state, action):
        return self.policy.get(state, {}).get(action, 0)

    def choose_action(self, state):
        # 返回当前状态下的动作
        actions = list(self.policy.get(state, {}).keys())
        pros = list(self.policy.get(state, {}).values())
        return np.random.choice(actions, p=pros)

    def update_policy(self, state, probabilities):
        # 更新某个状态的策略
        self.policy[state] = probabilities

    def print(self):
        ans = {}
        for state in self.states:
            val = -1
            best_action = None
            for action in self.actions:
                if self.get_pro(state, action) >= val:
                    best_action = action
                    val = self.get_pro(state, action)
            ans[state] = best_action
        return ans


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

    policy.update_policy(current_state, {"move":0.1, "stay":0.9})
    for _ in range(5):
        action = policy.choose_action(current_state)
        print(f"{_} state {current_state}, action: {action}")