import torch
import torch.optim as optim
import numpy as np
from agents.model import QNetwork
from agents.replay_buffer import ReplayBuffer
import torch.nn.functional as F
import os
import config
from pathlib import Path
from typing import Any
import gzip


class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, lr=1e-3, target_update=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.memory = ReplayBuffer(buffer_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        self.target_update = target_update
        self.epsilon_start = config.EPSILON_START
        self.epsilon_final = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon = config.EPSILON_START
        self.step_count = 0
        self.episode_count = 0
        self.actual_episode_count = 0

    def start_episode(self):
        """
        Call this exactly ONCE at the very beginning of each new episode (episode n).
        It computes self.epsilon = ε(n) based on the *current* episode_count,
        then increments episode_count so that next time we move to ε(n+1).
        """
        if self.actual_episode_count == 0:
            self.epsilon = 1
        elif self.actual_episode_count >= 500:
            self.epsilon = (
                self.epsilon_final
                + (self.epsilon_start - self.epsilon_final)
                * np.exp(-self.episode_count / self.epsilon_decay)
            )
            self.episode_count += 1
        self.actual_episode_count += 1
        

    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training:
            self.step_count += 1
            if np.random.random() < self.epsilon:
                return np.random.randint(self.action_size)
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
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
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
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            except Exception:
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_final)
            self.step_count = checkpoint.get('step_count', 0)
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")
            
    def export_script(self) -> torch.jit.ScriptModule:
        """Return a TorchScript version of the model for standalone use."""
        # Script only the network module
        scripted = torch.jit.script(self.net)
        return scripted

    def save_scripted(self, filepath: str, compress: bool = True, compresslevel: int = 9):
        """Convenience: export to TorchScript and save in one step."""
        scripted = self.export_script()
        DQNAgent.save_script(scripted, filepath, compress=compress, compresslevel=compresslevel)

    @staticmethod
    def save_script(scripted_module: torch.jit.ScriptModule, filepath: str, compress: bool = True, compresslevel: int = 9):
        """Save a TorchScript module to disk, optionally gzipped."""
        parent = Path(filepath).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        if compress:
            # Save to a temporary file then gzip-compress
            tmp_path = filepath + ".tmp"
            scripted_module.save(tmp_path)
            with gzip.open(filepath, 'wb', compresslevel=compresslevel) as f_out:
                with open(tmp_path, 'rb') as f_in:
                    f_out.write(f_in.read())
            # remove temporary file
            Path(tmp_path).unlink()
        else:
            scripted_module.save(filepath)

    @staticmethod
    def load_script(filepath: str, map_location: Any = 'cpu', compress: bool = True) -> torch.jit.ScriptModule:
        """Load a TorchScript module (gzipped or raw) for inference without class code."""
        if compress:
            with gzip.open(filepath, 'rb') as f:
                scripted = torch.jit.load(f, map_location=map_location)
        else:
            scripted = torch.jit.load(filepath, map_location=map_location)
        return scripted



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
        self.action_weights = [0.2, 0.2, 0.1, 0.5]    

    def act(self, state, training=True):
        return np.random.choice(self.action_size, p=self.action_weights)
