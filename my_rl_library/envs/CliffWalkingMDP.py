from my_rl_library.envs.MDP import MDP
import numpy as np

class CliffWalkingMDP(MDP):
    def __init__(self, m, n):
        self.m, self.n = m, n
        self.states = [(x, y) for x in range(m) for y in range(n)]
        self.actions = ['left ', 'right', 'up   ', 'down ']
        self.transition_probs = {
            (x, y): {action: self.gen_tran_probs((x, y), action)
                     for action in self.actions} for x in range(1, m) for y in range(n)
        }
        self.transition_probs[(0, 0)] = {
            'left ': {(0, 0): 1.0},
            'right': {(0, 1): 1.0},
            'down ': {(0, 0): 1.0},
            'up   ': {(1, 0): 1.0}, }
        self.rewards = {(x, y): -1 for x in range(m) for y in range(n)}
        for y in range(1, n - 1):
            self.rewards[0, y] = -100
        self.rewards[(0, n-1)] = 10
        super().__init__(self.states, self.actions, self.transition_probs, self.rewards)

    def gen_tran_probs(self, state, action):
        # 定义动作
        if action == 'down ':  # 下
            new_state = (max(state[0] - 1, 0), state[1])
        elif action == 'up   ':  # 上
            new_state = (min(state[0] + 1, self.m - 1), state[1])
        elif action == 'left ':  # 左
            new_state = (state[0], max(state[1] - 1, 0))
        elif action == 'right':  # 右
            new_state = (state[0], min(state[1] + 1, self.n - 1))
        else:
            raise ValueError("无效的动作")
        return {new_state: 1.0}

    def get_reward(self, state, action=None):
        return self.rewards.get(state, 0)

    def step(self, state, action):
        next_states_probs = self.get_next_states_probs(state, action)
        next_states = list(next_states_probs.keys())
        probs = list(next_states_probs.values())
        reward = self.get_reward(state, action)
        if sum(probs) == 0:
            return state, reward, True
        next_state = np.random.choice(list(map(lambda x: str(x[0]) + " " + str(x[1]), next_states)), p=probs)
        next_state = tuple(map(int, next_state.split()))
        return next_state, reward, False

if __name__ == "__main__":
    # 模拟一次随机游走
    mdp = CliffWalkingMDP(4, 10)
    # mdp.show()
    current_state = (0, 0)  # 从状态 2 开始
    done = False
    while not done:
        action = np.random.choice(mdp.actions)  # 随机选择动作
        next_state, reward, done = mdp.step(current_state, action)
        print(f"Current State: {current_state}, Action: {action}, Next State: {next_state}, Reward: {reward}")
        current_state = next_state
