import torch
import torch.optim as optim
import numpy as np
from agents.model import QNetwork
from agents.replay_buffer import ReplayBuffer
import torch.nn.functional as F
import os

class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=32,
                 gamma=0.99, lr=1e-3, target_update=1000,
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=100000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-Networks
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.step_count = 0

    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training:
            # Update epsilon
            self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                          np.exp(-1. * self.step_count / self.epsilon_decay)
            self.step_count += 1
            
            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                return np.random.randint(self.action_size)
        
        # Greedy action selection
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def step(self, state, action, reward, next_state, done):
        """Save experience and learn if enough samples available"""
        self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def learn(self, experiences):
        """Update Q-network using batch of experiences"""
        states, actions, rewards, next_states, dones = experiences
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filepath):
        """Save model state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)

    def load(self, filepath):
        """Load model state"""
        if os.path.exists(filepath):
            try:
                # Try loading with weights_only=True first (safer)
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            except Exception:
                # Fall back to weights_only=False for older model files
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_final)
            self.step_count = checkpoint.get('step_count', 0)
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")


class RandomAgent:
    """Random baseline agent for comparison"""
    def __init__(self, action_size):
        self.action_size = action_size
    
    def act(self, state, training=True):
        return np.random.randint(self.action_size)


class HeuristicAgent:
    """Simple heuristic agent that prefers certain actions"""
    def __init__(self, action_size):
        self.action_size = action_size
        # Action preferences: [left, right, rotate, drop]
        self.action_weights = [0.2, 0.2, 0.1, 0.5]  # Prefer dropping
    
    def act(self, state, training=True):
        return np.random.choice(self.action_size, p=self.action_weights)
