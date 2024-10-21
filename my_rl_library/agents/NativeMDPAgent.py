from my_rl_library.envs.RandomWalkMDP import RandomWalkMDP
from my_rl_library.algs.Policy import Policy

import numpy as np


class Agent:
    def __init__(self, mdp, gamma=1.0):
        self.mdp = mdp
        self.policy = Policy(mdp.states, mdp.actions)
        self.value_function = {state: 0 for state in mdp.states}
        self.gamma = gamma

    def get_pro(self, state, action):
        return self.policy.get_pro(state, action)

    def update_policy(self, state, probabilities):
        # 更新某个状态的策略
        self.policy.update_policy(state, probabilities)

    def evaluate_policy(self, num_episodes=1000):
        for state in self.mdp.states:
            total_reward = 0
            for _ in range(num_episodes):
                gamma = 1
                current_state = state
                done = False
                while not done:
                    action = self.policy.choose_action(current_state)
                    next_state, reward, done = self.mdp.step(current_state, action)
                    assert state != next_state or done, print(current_state, action, next_state, reward, done)
                    # print(current_state, action, next_state, reward, done)
                    total_reward += reward * gamma
                    gamma *= self.gamma
                    current_state = next_state
                    if current_state == 'terminal_state':  # 终止状态
                        done = True
            self.value_function[state] = total_reward / num_episodes
            print(self.value_function)

    def improve_policy(self):
        for state in self.mdp.states:
            best_action = self.mdp.actions[0]
            best_value = float('-inf')
            for action in self.mdp.actions:
                action_value = sum(
                    self.mdp.get_reward(state, action) + self.gamma * self.value_function[next_state] * prob
                    for next_state, prob in self.mdp.transition_probs.get(state, {}).get(action, {}).items()
                )
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            self.policy.policy[state] = np.array([0.0 if action == best_action else 1.0 for action in self.mdp.actions])

    def run(self, iterations=10):
        for _ in range(iterations):
            self.evaluate_policy()
            self.improve_policy()
            print(_, end=" iteration: ")
            self.policy.print()



if __name__ == "__main__":
    mdp = RandomWalkMDP(8)

    agent = Agent(mdp, 0.9)
    agent.run()

    print("Final Policy:", agent.policy.policy)
    print("Value Function:", agent.value_function)
