import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

    def show(self):
        # 创建图
        G = nx.DiGraph()

        # 添加节点和边
        for state in self.states:
            G.add_node(state)
            for action in self.actions:
                for next_state, prob in self.get_next_states_probs(state, action).items():
                    G.add_edge(state, next_state, weight=prob, action=action)

        # 设置图形位置
        pos = nx.spring_layout(G)

        fig = plt.figure()
        plt.clf()
        nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
        # edge_labels = {(u, v): f"{d['weight']}\n{d['action']}" for u, v, d in G.edges(data=True)}
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()

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
    mdp.show()

    pass