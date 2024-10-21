
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

class MDP_Env():
    def __init__(self, mdp, agent):
        self.mdp = mdp
        self.agent = agent

        self.reset()

        # 值函数V, Q
        self.state_value = {state: 0.0 for state in self.mdp.states}
        self.state_action_value = {state: {action: 0.0 for action in self.mdp.actions} for state in self.mdp.states}

    def reset(self):
        # SARS
        self.SARS = {}
        # episode
        self.episode = []
        self.cur_state = self.mdp.start_state
        return self.cur_state

    def calculate_V_func(self, epsilon=1e-4, max_T=-1):
        if self.agent.gamma == 1:
            max_T = 500
        # V(s) = E(G_t) = E(g_0+g_1+...)
        # s, a, r, s'
        for cur_state in self.mdp.states:
            state_pros = {state: (1.0 if state == cur_state else 0.0) for state in self.mdp.states}
            gamma = 1.0
            ans = 0.0
            i = 0
            while True:
                # print(state_pros)
                # 计算 E [R_t] = P [S_t] * P [A_t | S_t] * R
                new_pros = {state: 0.0 for state in self.mdp.states}
                E_R_t = 0.0
                for state in self.mdp.states:
                    pro_s = state_pros.get(state, 0.0)
                    for action in self.mdp.actions:
                        pro_a = agent.get_pro(state, action)
                        E_R_t += pro_s * pro_a * self.mdp.get_reward(state, action)
                        # print(state, action, pro_s, pro_a, self.mdp.get_reward(state, action))

                        probs = self.mdp.get_next_states_probs(state, action)
                        for next_state, val in probs.items():
                            new_pros[next_state] += val * pro_a * pro_s
                # print("ER:", E_R_t)
                if max_T != -1 and i > max_T:
                    break
                elif max_T == -1 and gamma < epsilon:
                    break
                ans += gamma * E_R_t
                state_pros = new_pros
                gamma *= self.agent.gamma
                i += 1
            self.state_value[cur_state] = ans

    def epsilon_greedy(self, epsilon=1e-1):
        probabilities = {}
        states_num = len(self.mdp.states)
        actions_num = len(self.mdp.actions)


        for state in self.mdp.states:
            tar = -np.Inf
            best_action = None
            probabilities = {action: epsilon / actions_num for action in self.mdp.actions}

            for action in self.mdp.actions:
                reward = self.mdp.get_reward(state, action)
                next_states = list(self.mdp.get_next_states_probs(state, action).keys())
                probs = list(self.mdp.get_next_states_probs(state, action).values())
                V_next = [probs * self.state_value[next_state] for next_state, probs in list(zip(next_states, probs))]
                # print(V_next)
                V_next = sum(V_next) / len(V_next) if len(V_next) != 0 else 0

                new_tar = reward + V_next
                best_action = action if new_tar >= tar else best_action
                tar = max(tar, new_tar)
            probabilities[best_action] += 1 - epsilon
            self.agent.update_policy(state, probabilities)


    def evaluate_policy(self, epsilon=1e-4, max_T=-1):
        self.calculate_V_func(epsilon, max_T)

    def improve_policy(self, epsilon=1e-1):
        self.epsilon_greedy(epsilon)
    def step(self):
        state = self.cur_state
        action = self.agent.policy.choose_action(state)
        next_state, reward, done = self.mdp.step(state, action)

        self.cur_state = next_state
        # print(state, action, next_state, reward, done)
        SARS = {}
        SARS["state"] = state
        SARS["action"] = action
        SARS["reward"] = reward
        SARS["next_state"] = next_state
        # 不能直接导入self.SARS
        self.SARS = SARS
        self.episode.append(SARS)
        return next_state, reward, done

    def animation(self, frames=60):
        # 创建图
        G = nx.DiGraph()

        # 添加节点和边
        for state in self.mdp.states:
            G.add_node(state)
            for action in self.mdp.actions:
                for next_state, prob in self.mdp.get_next_states_probs(state, action).items():
                    G.add_edge(state, next_state, weight=prob, action=action)

        # 设置图形位置
        pos = nx.planar_layout(G)

        # 创建动画更新函数
        def update(frame):
            plt.clf()
            nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
            # edge_labels = {(u, v): f"{d['weight']}\n{d['action']}" for u, v, d in G.edges(data=True)}
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            nx.draw_networkx_nodes(G, pos, nodelist=[self.cur_state], node_color='orange')
            # nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, width=4, alpha=1.0, edge_color='orange')
            # 高亮当前转移
            next_state, reward, done = self.step()
            plt.annotate('', xy=pos[self.SARS["next_state"]], xycoords='data',
                         xytext=pos[self.SARS["state"]], textcoords='data',
                         arrowprops=dict(arrowstyle="->", lw=2, color='red'))
            # print(self.SARS)
            if done:
                ani.event_source.stop()
                # plt.close(fig)
        # 创建动画
        fig = plt.figure()
        ani = FuncAnimation(fig, update, frames=frames, repeat=False)
        # 保存为视频文件
        # ani.save('mdp_transition.mp4', writer='ffmpeg', fps=1)
        plt.show()

    def run_episode(self, animation=False):
        self.reset()
        if animation:
            self.animation()
            return

        while True:
            s, r, done = self.step()
            if done:
                break

    def train(self, num_episodes=100, animation=False):
        return_list = []  # 记录每一条序列的回报
        for i in range(10):  # 显示10个进度条
            # tqdm的进度条功能
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                    self.run_episode(animation)
                    self.evaluate_policy()
                    self.improve_policy()

                    reward_lst = [SARS.get("reward", 0) for SARS in self.episode]
                    # print(reward_lst)
                    episode_return = sum(reward_lst)
                    return_list.append(episode_return)
                    if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                            'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                            'return':
                            '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)

if __name__ == "__main__":
    # from my_rl_library.envs.RandomWalkMDP import RandomWalkMDP
    from my_rl_library.envs.CliffWalkingMDP import CliffWalkingMDP
    from my_rl_library.agents.NativeMDPAgent import Agent
    # 创建 MDP 实例
    # mdp = RandomWalkMDP(10)
    mdp = CliffWalkingMDP(4, 6)
    agent = Agent(mdp)
    env = MDP_Env(mdp, agent)
    # mdp.show()

    for _ in range(100):
        s, r, done = env.step()
        if done:
            break

    env.train()

    policy = env.agent.policy.print()
    print()
    print()
    for a in range(env.mdp.m):
        for b in range(env.mdp.n):
            action = policy[(a, b)][0]
            print(action, end='')
        print()

    # plt.show()
    # env.animation()
