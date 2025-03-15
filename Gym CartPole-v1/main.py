import random

import gym
import numpy as np
import tensorflow as tf
from keras import layers, models
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

env = gym.make("CartPole-v1", render_mode="human")

states = env.observation_space.shape[0]
actions = env.action_space.n

model = models.Sequential([
    layers.Input(shape=(1, states)),
    layers.Flatten(),
    layers.Dense(24, activation="relu"),
    layers.Dense(24, activation="relu"),
    layers.Dense(actions, activation="relu"),
])

env.close()
