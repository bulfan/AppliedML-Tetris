import torch
import torch.optim as optim
import numpy as np
from agents.model import QNetwork
from agents.replay_buffer import ReplayBuffer
import torch.nn.functional as F

class DQNAgent:
    def __init__(self, env, buffer_size=10000, batch_size=32,
                 gamma=0.99, lr=1e-5, target_update=1000,
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=100000):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = env.action_space
        self.policy_net = QNetwork(env.height, env.width, self.n_actions).to(self.device)
        self.target_net = QNetwork(env.height, env.width, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.buffer = ReplayBuffer(buffer_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.epsilon_start, self.epsilon_final, self.epsilon_decay = (
            epsilon_start, epsilon_final, epsilon_decay)
        self.step_count = 0

    def select_action(self, state):
        eps = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
              np.exp(-1. * self.step_count / self.epsilon_decay)
        self.step_count += 1

        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)
        else:
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                qvals = self.policy_net(state_t)
            return qvals.argmax().item()

    def optimize(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)

        # current Q values
        q_vals = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # target Q
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            q_target = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
