import random

import gym
from keras import layers, models
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

env = gym.make("CartPole-v1", render_mode="human")

steps = 10

for step in range(1, steps+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0, 1])
        _, reward, done, _ = env.step(action)
        score += reward
        env.render()

    print(f"Step: {step}, score: {score}")

env.close()
