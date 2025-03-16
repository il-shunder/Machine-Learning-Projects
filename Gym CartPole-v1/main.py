import gym
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

env = gym.make("CartPole-v1")

states = env.observation_space.shape[0]
actions = env.action_space.n

model = models.Sequential([
    layers.Input(shape=(1, states)),
    layers.Flatten(),
    layers.Dense(24, activation="relu"),
    layers.Dense(24, activation="relu"),
    layers.Dense(actions, activation="relu"),
])

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

agent.compile(optimizers.Adam(0.001), metrics=["mae"])
agent.fit(env, nb_steps=10000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=10, visualize=True)

print(np.mean(results.history["episode_reward"]))

env.close()
