import numpy as np


class MDP:
    def __init__(self, states, actions, transition_probs, rewards):
        # 此处我们没有建模gamma, 而是将其放置在策略中进行考虑
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs  # Dict: {state: {action: {next_state: probability}}}
        self.rewards = rewards  # Dict: {(state, action): reward}

    def get_transition_prob(self, state, action, next_state):
        return self.transition_probs.get(state, {}).get(action, {}).get(next_state, 0)

    def get_reward(self, state, action):
        return self.rewards.get((state, action), 0)

    def step(self, state, action):
        next_states = list(self.transition_probs.get(state, {}).get(action, {}).keys())
        probs = list(self.transition_probs.get(state, {}).get(action, {}).values())
        reward = self.get_reward(state, action)
        if len(probs) == 0:
            return state, reward, True
        next_state = np.random.choice(next_states, p=probs)
        done = False
        return next_state, reward, done

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
    next_state, reward, _ = mdp.step('A', 'move')
    print(f"Next State: {next_state}, Reward: {reward}")
    next_state, reward, _ = mdp.step('A', 'stay')
    print(f"Next State: {next_state}, Reward: {reward}")