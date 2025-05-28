import unittest
import numpy as np
import torch
import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agents.dqn_agent import DQNAgent, RandomAgent, HeuristicAgent
from agents.model import QNetwork
class TestDQNAgent(unittest.TestCase):
    """Test cases for the DQNAgent class"""
    def setUp(self):
        """Set up test fixtures"""
        self.state_size = 220        
        self.action_size = 4
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            buffer_size=1000,
            batch_size=16,
            epsilon_decay=1000
        )
    def test_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertIsInstance(self.agent.q_network, QNetwork)
        self.assertIsInstance(self.agent.target_network, QNetwork)
        self.assertEqual(
            list(self.agent.q_network.parameters())[0].shape,
            list(self.agent.target_network.parameters())[0].shape
        )
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertEqual(self.agent.batch_size, 16)
        self.assertEqual(self.agent.epsilon_start, 1.0)
        self.assertEqual(self.agent.epsilon_final, 0.01)
    def test_act_training_mode(self):
        """Test action selection in training mode"""
        state = np.random.random(self.state_size)
        actions = []
        for _ in range(10):
            action = self.agent.act(state, training=True)
            actions.append(action)
            self.assertIn(action, range(self.action_size))
            self.assertIsInstance(action, int)
        self.assertGreater(len(set(actions)), 1)
    def test_act_evaluation_mode(self):
        """Test action selection in evaluation mode"""
        state = np.random.random(self.state_size)
        actions = []
        for _ in range(5):
            action = self.agent.act(state, training=False)
            actions.append(action)
            self.assertIn(action, range(self.action_size))
        unique_actions = set(actions)
        self.assertLessEqual(len(unique_actions), self.action_size)
        initial_epsilon = self.agent.epsilon
        self.agent.act(state, training=False)
        self.assertEqual(self.agent.epsilon, initial_epsilon)
    def test_epsilon_decay(self):
        """Test epsilon decay over time"""
        initial_epsilon = self.agent.epsilon
        state = np.random.random(self.state_size)
        for _ in range(100):
            self.agent.act(state, training=True)
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_final)
    def test_step_and_memory(self):
        """Test experience storage and learning trigger"""
        state = np.random.random(self.state_size)
        action = 0
        reward = 1.0
        next_state = np.random.random(self.state_size)
        done = False
        initial_memory_size = len(self.agent.memory)
        self.agent.step(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.memory), initial_memory_size + 1)
    def test_learn_with_sufficient_data(self):
        """Test learning when enough experiences are available"""
        for _ in range(self.agent.batch_size + 5):
            state = np.random.random(self.state_size)
            action = np.random.randint(self.action_size)
            reward = np.random.random()
            next_state = np.random.random(self.state_size)
            done = np.random.choice([True, False])
            self.agent.memory.add(state, action, reward, next_state, done)
        initial_weights = self.agent.q_network.state_dict()['network.0.weight'].clone()
        experiences = self.agent.memory.sample(self.agent.batch_size)
        self.agent.learn(experiences)
        new_weights = self.agent.q_network.state_dict()['network.0.weight']
        self.assertFalse(torch.equal(initial_weights, new_weights))
    def test_target_network_update(self):
        """Test target network update"""
        initial_target_weights = self.agent.target_network.state_dict()['network.0.weight'].clone()
        with torch.no_grad():
            self.agent.q_network.network[0].weight += 0.1
        self.agent.step_count = self.agent.target_update
        for _ in range(self.agent.batch_size + 1):
            state = np.random.random(self.state_size)
            action = np.random.randint(self.action_size)
            reward = np.random.random()
            next_state = np.random.random(self.state_size)
            done = False
            self.agent.step(state, action, reward, next_state, done)
        new_target_weights = self.agent.target_network.state_dict()['network.0.weight']
        self.assertFalse(torch.equal(initial_target_weights, new_target_weights))
    def test_save_and_load(self):
        """Test model saving and loading"""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            self.agent.epsilon = 0.5
            self.agent.step_count = 1000
            self.agent.save(tmp_path)
            new_agent = DQNAgent(self.state_size, self.action_size)
            new_agent.load(tmp_path)
            self.assertEqual(new_agent.epsilon, 0.5)
            self.assertEqual(new_agent.step_count, 1000)
            original_weights = self.agent.q_network.state_dict()['network.0.weight']
            loaded_weights = new_agent.q_network.state_dict()['network.0.weight']
            torch.testing.assert_close(original_weights, loaded_weights)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file"""
        self.agent.load("nonexistent_file.pth")
        self.assertEqual(self.agent.epsilon, self.agent.epsilon_start)
        self.assertEqual(self.agent.step_count, 0)
class TestRandomAgent(unittest.TestCase):
    """Test cases for the RandomAgent class"""
    def setUp(self):
        """Set up test fixtures"""
        self.action_size = 4
        self.agent = RandomAgent(self.action_size)
    def test_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.action_size, self.action_size)
    def test_act(self):
        """Test action selection"""
        state = np.random.random(220)
        actions = []
        for _ in range(100):
            action = self.agent.act(state)
            actions.append(action)
            self.assertIn(action, range(self.action_size))
            self.assertIsInstance(action, (int, np.integer))
        unique_actions = set(actions)
        self.assertGreaterEqual(len(unique_actions), 2)    
    def test_act_training_and_evaluation(self):
        """Test that training flag doesn't affect random agent"""
        state = np.random.random(220)
        action1 = self.agent.act(state, training=True)
        action2 = self.agent.act(state, training=False)
        self.assertIn(action1, range(self.action_size))
        self.assertIn(action2, range(self.action_size))
class TestHeuristicAgent(unittest.TestCase):
    """Test cases for the HeuristicAgent class"""
    def setUp(self):
        """Set up test fixtures"""
        self.action_size = 4
        self.agent = HeuristicAgent(self.action_size)
    def test_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertEqual(len(self.agent.action_weights), self.action_size)
        self.assertAlmostEqual(sum(self.agent.action_weights), 1.0, places=5)
    def test_act(self):
        """Test action selection"""
        state = np.random.random(220)
        actions = []
        for _ in range(1000):            
            action = self.agent.act(state)
            actions.append(action)
            self.assertIn(action, range(self.action_size))
            self.assertIsInstance(action, (int, np.integer))
        action_counts = [actions.count(i) for i in range(self.action_size)]
        action_probs = [count / len(actions) for count in action_counts]
        self.assertEqual(np.argmax(action_probs), 3)
        self.assertEqual(len(set(actions)), self.action_size)
    def test_action_weights(self):
        """Test action weight configuration"""
        self.assertEqual(self.agent.action_weights[3], 0.5)        
        self.assertEqual(self.agent.action_weights[0], 0.2)        
        self.assertEqual(self.agent.action_weights[1], 0.2)        
        self.assertEqual(self.agent.action_weights[2], 0.1)

class TestAgentIntegration(unittest.TestCase):
    """Integration tests for agents with environment"""
    def setUp(self):
        """Set up test fixtures"""
        self.state_size = 220
        self.action_size = 4
        self.dqn_agent = DQNAgent(self.state_size, self.action_size, buffer_size=100, batch_size=8)
        self.random_agent = RandomAgent(self.action_size)
        self.heuristic_agent = HeuristicAgent(self.action_size)
    def test_agents_with_environment_states(self):
        """Test that all agents work with environment-like states"""
        state = np.random.randint(0, 8, size=self.state_size).astype(np.float32)
        dqn_action = self.dqn_agent.act(state)
        random_action = self.random_agent.act(state)
        heuristic_action = self.heuristic_agent.act(state)
        for action in [dqn_action, random_action, heuristic_action]:
            self.assertIn(action, range(self.action_size))
    def test_dqn_agent_learning_loop(self):
        """Test DQN agent in a simple learning loop"""
        for episode in range(3):
            state = np.random.random(self.state_size)
            for step in range(10):
                action = self.dqn_agent.act(state, training=True)
                reward = np.random.random() - 0.5                
                next_state = np.random.random(self.state_size)
                done = step == 9                
                self.dqn_agent.step(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
        self.assertGreater(len(self.dqn_agent.memory), 0)
        self.assertLess(self.dqn_agent.epsilon, self.dqn_agent.epsilon_start)
if __name__ == '__main__':
    unittest.main() 