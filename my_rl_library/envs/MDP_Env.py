
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation


class MDP_Env():
    def __init__(self, mdp, agent):
        self.mdp = mdp
        self.agent = agent
        self.cur_state = self.mdp.states[4]

    def step(self):
        state = self.cur_state
        action = self.agent.policy.choose_action(state)
        next_state, reward, done = self.mdp.step(state, action)

        self.cur_state = next_state
        print(state, action, next_state, reward, done)
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
        pos = nx.spring_layout(G, iterations=800)

        # 创建动画更新函数
        def update(frame):
            plt.clf()
            nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
            edge_labels = {(u, v): f"{d['weight']}\n{d['action']}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            nx.draw_networkx_nodes(G, pos, nodelist=[self.cur_state], node_color='orange')
            # nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, width=4, alpha=1.0, edge_color='orange')
            # 高亮当前转移
            cur_state, next_state = 0, 0
            plt.annotate('', xy=pos[next_state], xycoords='data',
                         xytext=pos[cur_state], textcoords='data',
                         arrowprops=dict(arrowstyle="->", lw=2, color='red'))

            next_state, reward, done = self.step()
            if done:
                ani.event_source.stop()
                # plt.close(fig)

        # 创建动画
        fig = plt.figure()
        ani = FuncAnimation(fig, update, frames=frames, repeat=False)

        # 保存为视频文件
        # ani.save('mdp_transition.mp4', writer='ffmpeg', fps=1)

        plt.show()


if __name__ == "__main__":
    from my_rl_library.envs.RandomWalkMDP import RandomWalkMDP
    from my_rl_library.agents.NativeMDPAgent import Agent
    # 创建 MDP 实例
    mdp = RandomWalkMDP(8)
    agent = Agent(mdp)
    env = MDP_Env(mdp, agent)

    env.animation()
