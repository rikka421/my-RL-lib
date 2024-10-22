

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

```python
def __init__(self, bandit, epsilon=0.01, init_prob=1.0, init_N=0):...
```

 - 对于epsilon, 等于0时退化为greedy; 等于1时退化为随机策略;
 - 对于init_prob, 大于1时成为乐观初始化方法;
 - 对于init_N, 主要用于纯greedy时的前期探索, greedy时取`3*K`, 对于非纯greedy的方法, 一般取0

可以实现
 1. 贪心策略和$\epsilon$-greedy策略
 2. 乐观初始化

基于不确定性的度量


Thompson Sampling方法

### BanditMainEnv
[BanditMainEnv.py](..%2F..%2Fenvs%2FBanditEnv%2FBanditMainEnv.py)

此处编写的
```python
def __init__(self, env, agents):...
```

会接受一个统一的环境env, 同时接受一系列的agent, 这些agent都各自运行在env上. 

在后面的编写中, 可能会有一个env中包含多个anent的情况, 它们都统一到架构mainEnv中. 

通过mainEnv.run的调用, 我们可以方便地完成多个agent在各自环境;
一个agent在多个环境; 多个agent在同一个环境的集成运行

### 一些实验结论

 - 仅对于epsilon-greedy算法, 当T=5000, K=10时, epsilon=1e-4时的结果总是优于参数更大时的结果
 - `epsilon = 1/t`时, 算法表现与`e=1e-4`大致相同, 同时都优于`e=0.01`的设置




