import os
import gymnasium as gym
import time
done_id = """
CartPole-v1
MountainCar-v0
MountainCarContinuous-v0
Pendulum-v1
Acrobot-v1
#CartPoleJax-v0  
CartPoleJax-v1
"""

name = "Humanoid"
id_str = """
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
#Reacher-v2
Reacher-v4
#Pusher-v2
Pusher-v4
#InvertedPendulum-v2
InvertedPendulum-v4
#InvertedDoublePendulum-v2
InvertedDoublePendulum-v4
#HalfCheetah-v2
#HalfCheetah-v3
HalfCheetah-v4
#Hopper-v2
#Hopper-v3
Hopper-v4
#Swimmer-v2
#Swimmer-v3
Swimmer-v4
#Walker2d-v2
#Walker2d-v3
Walker2d-v4
#Ant-v2
#Ant-v3
Ant-v4
#Humanoid-v2
#Humanoid-v3
Humanoid-v4
#HumanoidStandup-v2
HumanoidStandup-v4
#GymV21Environment-v0
GymV26Environment-v0


InvertedPendulum
InvertedDoublePendulum
Reacher
Pusher
HalfCheetah
Hopper
Walker2d
Swimmer
Ant
Swimmer
Humanoid
HumanoidStandup
"""

mujoco_envs = """
"""

max_T = 50000


# for name in mujoco_envs.split("\n")[1:-1]:

env = gym.make(name, render_mode="human")
observation, info = env.reset()

episode_over = False
ti = 0
while True:
    if ti % 100 == 0:
        print(env.action_space)

    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated or (ti > max_T)
    ti += 1

env.close()

print(name, "ok")
time.sleep(1)