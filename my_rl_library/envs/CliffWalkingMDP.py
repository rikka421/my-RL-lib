from my_rl_library.envs.MDP import MDP
import numpy as np

class CliffWalkingMDP(MDP):
    def __init__(self, m, n):
        self.m, self.n = m, n
        self.states = [str(x) + " " + str(y) for x in range(m) for y in range(n)]
        self.actions = ['left ', 'right', 'up   ', 'down ']
        self.transition_probs = {
            str(x) + " " + str(y): {action: self.gen_tran_probs(str(x) + " " + str(y), action)
                                    for action in self.actions} for x in range(1, m) for y in range(n)
        }
        self.transition_probs["0 0"] = {
            'left ': {"0 0": 1.0},
            'right': {"0 1": 1.0},
            'up   ': {"0 0": 1.0},
            'down ': {"1 0": 1.0}, }
        self.rewards = {"0 " + str(y): -100 for y in range(1, n)}
        self.rewards["0 " + str(n-1)] = 10
        super().__init__(self.states, self.actions, self.transition_probs, self.rewards)

    def gen_tran_probs(self, state_str, action):
        state = tuple(map(int, state_str.split()))
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
        new_state = tuple(map(str, new_state))
        new_state_str = " ".join(new_state)
        return {new_state_str: 1.0}

    def get_reward(self, state, action=None):
        return self.rewards.get(state, 0)


if __name__ == "__main__":
    # 模拟一次随机游走
    mdp = CliffWalkingMDP(4, 10)
    # mdp.show()
    current_state = "0 0"  # 从状态 2 开始
    done = False
    while not done:
        action = np.random.choice(mdp.actions)  # 随机选择动作
        next_state, reward, done = mdp.step(current_state, action)
        print(f"Current State: {current_state}, Action: {action}, Next State: {next_state}, Reward: {reward}")
        current_state = next_state
