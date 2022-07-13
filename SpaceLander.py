import time

from tutorial import Agent
import numpy as np
import gym
import tensorflow as tf
import Box2D

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    lr = .01
    n_games = 500
    agent = Agent(gamma=.99, eps=1.0, lr=lr, input_dims=env.observation_space.shape, n_actions=env.action_space.n, mem_size=1000000, batch_size=64, eps_end=.01, eps_dec=.001)
    scores = []
    eps_history = []
    i = 1
    while True:
        done = False
        score = 0
        observation = env.reset()
        while not done:
            print(observation)
            time.sleep(0.5)
            env.render()
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, new_observation, done)
            observation = new_observation
            agent.learn()
        eps_history.append(agent.eps)
        print('episode:', agent.tries, 'score %.3f' % score, 'epsilon %.3f' % agent.eps)
        # agent.dec_eps()
        agent.tries += 1

