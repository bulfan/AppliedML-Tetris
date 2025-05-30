import random
from collections import deque
import numpy as np
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def add(self, state, action, reward, next_state, done):
        """Alias for push method to match DQN agent interface"""
        self.push(state, action, reward, next_state, done)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(
            lambda x: np.array(x), zip(*batch))
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)
