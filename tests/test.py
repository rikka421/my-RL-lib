import numpy as np
import random
from collections import deque


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, transition, priority):
        self.buffer.append(transition)
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        # Calculate priority-based probabilities
        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Importance-sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize

        batch = [self.buffer[idx] for idx in indices]

        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        # Initialize the agent with Q network, target network, etc.
        self.replay_buffer = PrioritizedReplayBuffer(100000)
        self.batch_size = 32
        self.gamma = 0.99
        self.beta = 0.4
        self.alpha = 0.6

    def train(self):
        # Sample from prioritized experience replay buffer
        batch, indices, weights = self.replay_buffer.sample(self.batch_size, self.beta)

        # Get states, actions, rewards, next states from batch
        states, actions, rewards, next_states = zip(*batch)

        # Compute the TD errors (target)
        q_values_next = self.target_network(next_states)
        target_q = rewards + self.gamma * np.max(q_values_next, axis=1)

        # Compute the predicted Q-values
        q_values = self.q_network(states)
        q_pred = q_values[np.arange(self.batch_size), actions]

        # Compute the loss with importance sampling weights
        errors = target_q - q_pred
        loss = np.mean(weights * np.square(errors))  # Loss with IS correction

        # Backpropagate to update the Q network
        # (Use optimizer and compute gradients)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in the replay buffer
        new_priorities = np.abs(errors) + 1e-5
        self.replay_buffer.update_priorities(indices, new_priorities)
