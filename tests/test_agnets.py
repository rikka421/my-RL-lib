 
import numpy as np

class MDP:
    def __init__(self, states, actions, transition_probs, rewards):
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs
        self.rewards = rewards

    def get_reward(self, state, action):
        return self.rewards.get((state, action), 0)

    def step(self, state, action):
        next_states = list(self.transition_probs[state][action].keys())
        probs = list(self.transition_probs[state][action].values())
        next_state = np.random.choice(next_states, p=probs)
        reward = self.get_reward(state, action)
        return next_state, reward

