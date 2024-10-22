

## 多臂老虎机问题. 

此处我们考虑每个手臂都服从伯努利分布. 


### Solver

基本的Solver框架在
[BernoulliBanditSolver.py](BernoulliBanditSolver.py)
中

### EpsilonGreedySolver
编写了
[EpsilonGreedySolver.py](EpsilonGreedySolver.py)

其中, 有三个超参数

    def __init__(self, bandit, epsilon=0.01, init_prob=1.0, init_N=0)

 - 对于epsilon, 等于0时退化为greedy; 等于1时退化为随机策略;
 - 对于init_prob, 大于1时成为乐观初始化方法;
 - 对于init_N, 主要用于纯greedy时的前期探索, greedy时取`3*K`, 对于非纯greedy的方法, 一般取0

可以实现
 1. 贪心策略和$\epsilon$-greedy策略
 2. 乐观初始化

基于不确定性的度量


Thompson Sampling方法