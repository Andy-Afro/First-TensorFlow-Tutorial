import numpy as np
import tensorflow as ts
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential

class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self. new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = new_state
        self.reward_memory[idx] = reward
        self.action_memory[idx] = action
        self.terminal_memory[idx] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential(
        [keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation=None)])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model

class Agent:
    def __init__(self, lr, gamma, n_actions, eps, batch_size, input_dims, eps_dec=.001, eps_end=.01, mem_size=1000000, fname='dpnmodel.h5'):
        self.actions = [i for i in range(n_actions)]
        self.gamma = gamma
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_min = eps_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)
        self.tries = 0

    def store_transition(self, state, action, reward, new_stae, done):
        self.memory.store_transition(state, action, reward, new_stae, done)

    def choose_action(self, observation):
        if np.random.random() < self.eps:
            action = np.random.choice(self.actions)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)

            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        else:
            states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

            q_eval = self.q_eval.predict(states)
            q_next = self.q_eval.predict(new_states)

            q_target = np.copy(q_eval)
            batch_idx = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_idx, actions] = rewards + self.gamma + np.max(q_next, axis=1)*dones

            self.q_eval.train_on_batch(states, q_target)
            self.dec_eps()

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)

    def dec_eps(self):
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min

