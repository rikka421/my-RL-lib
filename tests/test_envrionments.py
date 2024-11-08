import gymnasium as gym

# 列出所有可用环境
envs = gym.envs.registry
env_names = [env for env in envs]

print("可用的环境:")
for name in env_names:
    print(name)

id_str = """
CartPole-v0
CartPole-v1
MountainCar-v0
MountainCarContinuous-v0
Pendulum-v1
Acrobot-v1
CartPoleJax-v0
CartPoleJax-v1
PendulumJax-v0
LunarLander-v2
LunarLanderContinuous-v2
BipedalWalker-v3
BipedalWalkerHardcore-v3
CarRacing-v2
Blackjack-v1
FrozenLake-v1
FrozenLake8x8-v1
CliffWalking-v0
Taxi-v3
Jax-Blackjack-v0
Reacher-v2
Reacher-v4
Pusher-v2
Pusher-v4
InvertedPendulum-v2
InvertedPendulum-v4
InvertedDoublePendulum-v2
InvertedDoublePendulum-v4
HalfCheetah-v2
HalfCheetah-v3
HalfCheetah-v4
Hopper-v2
Hopper-v3
Hopper-v4
Swimmer-v2
Swimmer-v3
Swimmer-v4
Walker2d-v2
Walker2d-v3
Walker2d-v4
Ant-v2
Ant-v3
Ant-v4
Humanoid-v2
Humanoid-v3
Humanoid-v4
HumanoidStandup-v2
HumanoidStandup-v4
GymV21Environment-v0
GymV26Environment-v0
"""

print(id_str.split('\n'))



"""



Robot

Short Description

CartPoles

InvertedPendulum

MuJoCo version of the CartPole Environment (with Continuous actions)

InvertedDoublePendulum

2 Pole variation of the CartPole Environment

Arms

Reacher

2d arm with the goal of reaching an object

Pusher

3d arm with the goal of pushing an object to a target location

2D Runners

HalfCheetah

2d quadruped with the goal of running

Hopper

2d monoped with the goal of hopping

Walker2d

2d biped with the goal of walking

Swimmers

Swimmer

3d robot with the goal of swimming

Quarduped

Ant

3d quadruped with the goal of running

Humanoid Bipeds

Humanoid

3d humanoid with the goal of running

HumanoidStandup

3d humanoid with the goal of standing up
"""