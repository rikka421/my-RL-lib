from my_rl_library.envs.MDP import MDP
import numpy as np


class RandomWalkMDP(MDP):
    def __init__(self, n):
        # 定义状态、动作和奖励
        self.states = list(range(n))
        self.actions = ['left', 'right']
        self.states_num = n
        self.actions_num = len(self.actions)

        self.start_state = n // 2
        self.terminal_states = [0, n-1]
        self.transition_probs = None
        self.init_transition_probs()

        self.rewards = [0] * n
        self.rewards[n-1] = 1
        self.rewards = np.array(self.rewards)
        super().__init__(self.states_num, self.actions_num, self.transition_probs, self.rewards,
                         self.start_state, self.terminal_states)

    def init_transition_probs(self):
        self.transition_probs = np.zeros((self.states_num, self.actions_num, self.states_num), dtype=float)
        for i in range(self.states_num):
            for j in range(2):
                if i == 0:
                    continue
                elif i == self.states_num - 1:
                    continue
                else:
                    self.transition_probs[i, j, i+(j * 2 - 1)] = 1.0

        for s in self.terminal_states:
            self.transition_probs[s, :, s] = np.ones((self.actions_num))

    def get_reward(self, state, action=None):
        return self.rewards[state]


if __name__ == "__main__":
    # 模拟一次随机游走
    mdp = RandomWalkMDP(5)
    # print(mdp.transition_probs)
    print(mdp.rewards)

    s = mdp.reset()
    done = False
    while not done:
        action = np.random.choice([0, 1])
        ns, r, done = mdp.step(s, action)
        print(s, action, ns, r, done)
        s = ns