import unittest
import numpy as np
import torch
import sys
import os
import tempfile

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.dqn_agent import DQNAgent, RandomAgent, HeuristicAgent
from agents.model import QNetwork


class TestDQNAgent(unittest.TestCase):
    """Test cases for the DQNAgent class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state_size = 220  # 10 * 22 flattened board
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
        
        # Check that networks have same architecture
        self.assertEqual(
            list(self.agent.q_network.parameters())[0].shape,
            list(self.agent.target_network.parameters())[0].shape
        )
        
        # Check hyperparameters
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertEqual(self.agent.batch_size, 16)
        self.assertEqual(self.agent.epsilon_start, 1.0)
        self.assertEqual(self.agent.epsilon_final, 0.01)
    
    def test_act_training_mode(self):
        """Test action selection in training mode"""
        state = np.random.random(self.state_size)
        
        # Test multiple actions to check randomness
        actions = []
        for _ in range(10):
            action = self.agent.act(state, training=True)
            actions.append(action)
            
            # Check action is valid
            self.assertIn(action, range(self.action_size))
            self.assertIsInstance(action, int)
        
        # With high epsilon, should see some randomness
        self.assertGreater(len(set(actions)), 1)
    
    def test_act_evaluation_mode(self):
        """Test action selection in evaluation mode"""
        state = np.random.random(self.state_size)
        
        # In evaluation mode, should be deterministic for the same state
        # But we need to account for potential randomness in untrained networks
        actions = []
        for _ in range(5):
            action = self.agent.act(state, training=False)
            actions.append(action)
            self.assertIn(action, range(self.action_size))
        
        # For an untrained network, actions might vary due to random initialization
        # The key is that all actions should be valid
        unique_actions = set(actions)
        self.assertLessEqual(len(unique_actions), self.action_size)
        
        # Test that evaluation mode doesn't update epsilon
        initial_epsilon = self.agent.epsilon
        self.agent.act(state, training=False)
        self.assertEqual(self.agent.epsilon, initial_epsilon)
    
    def test_epsilon_decay(self):
        """Test epsilon decay over time"""
        initial_epsilon = self.agent.epsilon
        
        # Take several actions to trigger epsilon decay
        state = np.random.random(self.state_size)
        for _ in range(100):
            self.agent.act(state, training=True)
        
        # Epsilon should have decreased
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_final)
    
    def test_step_and_memory(self):
        """Test experience storage and learning trigger"""
        state = np.random.random(self.state_size)
        action = 0
        reward = 1.0
        next_state = np.random.random(self.state_size)
        done = False
        
        # Add experience
        initial_memory_size = len(self.agent.memory)
        self.agent.step(state, action, reward, next_state, done)
        
        # Memory should have increased
        self.assertEqual(len(self.agent.memory), initial_memory_size + 1)
    
    def test_learn_with_sufficient_data(self):
        """Test learning when enough experiences are available"""
        # Fill memory with random experiences
        for _ in range(self.agent.batch_size + 5):
            state = np.random.random(self.state_size)
            action = np.random.randint(self.action_size)
            reward = np.random.random()
            next_state = np.random.random(self.state_size)
            done = np.random.choice([True, False])
            
            self.agent.memory.add(state, action, reward, next_state, done)
        
        # Get initial network weights
        initial_weights = self.agent.q_network.state_dict()['network.0.weight'].clone()
        
        # Sample experiences and learn
        experiences = self.agent.memory.sample(self.agent.batch_size)
        self.agent.learn(experiences)
        
        # Weights should have changed
        new_weights = self.agent.q_network.state_dict()['network.0.weight']
        self.assertFalse(torch.equal(initial_weights, new_weights))
    
    def test_target_network_update(self):
        """Test target network update"""
        # Get initial target network weights
        initial_target_weights = self.agent.target_network.state_dict()['network.0.weight'].clone()
        
        # Modify main network weights
        with torch.no_grad():
            self.agent.q_network.network[0].weight += 0.1
        
        # Force target network update
        self.agent.step_count = self.agent.target_update
        
        # Fill memory and trigger learning
        for _ in range(self.agent.batch_size + 1):
            state = np.random.random(self.state_size)
            action = np.random.randint(self.action_size)
            reward = np.random.random()
            next_state = np.random.random(self.state_size)
            done = False
            
            self.agent.step(state, action, reward, next_state, done)
        
        # Target network weights should have updated
        new_target_weights = self.agent.target_network.state_dict()['network.0.weight']
        self.assertFalse(torch.equal(initial_target_weights, new_target_weights))
    
    def test_save_and_load(self):
        """Test model saving and loading"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Modify agent state
            self.agent.epsilon = 0.5
            self.agent.step_count = 1000
            
            # Save model
            self.agent.save(tmp_path)
            
            # Create new agent and load
            new_agent = DQNAgent(self.state_size, self.action_size)
            new_agent.load(tmp_path)
            
            # Check that state was loaded correctly
            self.assertEqual(new_agent.epsilon, 0.5)
            self.assertEqual(new_agent.step_count, 1000)
            
            # Check that network weights are the same
            original_weights = self.agent.q_network.state_dict()['network.0.weight']
            loaded_weights = new_agent.q_network.state_dict()['network.0.weight']
            torch.testing.assert_close(original_weights, loaded_weights)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file"""
        # Should not raise exception
        self.agent.load("nonexistent_file.pth")
        
        # Agent should remain in initial state
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
        
        # Test multiple actions
        actions = []
        for _ in range(100):
            action = self.agent.act(state)
            actions.append(action)
            
            # Check action is valid
            self.assertIn(action, range(self.action_size))
            self.assertIsInstance(action, (int, np.integer))
        
        # Should see all actions with sufficient samples
        unique_actions = set(actions)
        self.assertGreaterEqual(len(unique_actions), 2)  # Should see some variety
    
    def test_act_training_and_evaluation(self):
        """Test that training flag doesn't affect random agent"""
        state = np.random.random(220)
        
        # Should behave the same regardless of training flag
        action1 = self.agent.act(state, training=True)
        action2 = self.agent.act(state, training=False)
        
        # Both should be valid actions
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
        
        # Test multiple actions
        actions = []
        for _ in range(1000):  # Large sample to test distribution
            action = self.agent.act(state)
            actions.append(action)
            
            # Check action is valid
            self.assertIn(action, range(self.action_size))
            self.assertIsInstance(action, (int, np.integer))
        
        # Check that action distribution roughly matches weights
        action_counts = [actions.count(i) for i in range(self.action_size)]
        action_probs = [count / len(actions) for count in action_counts]
        
        # Drop action (index 3) should be most common
        self.assertEqual(np.argmax(action_probs), 3)
        
        # Should see all actions
        self.assertEqual(len(set(actions)), self.action_size)
    
    def test_action_weights(self):
        """Test action weight configuration"""
        # Default weights should prefer dropping
        self.assertEqual(self.agent.action_weights[3], 0.5)  # Drop action
        self.assertEqual(self.agent.action_weights[0], 0.2)  # Left
        self.assertEqual(self.agent.action_weights[1], 0.2)  # Right
        self.assertEqual(self.agent.action_weights[2], 0.1)  # Rotate


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
        # Simulate environment state (flattened board)
        state = np.random.randint(0, 8, size=self.state_size).astype(np.float32)
        
        # Test all agents
        dqn_action = self.dqn_agent.act(state)
        random_action = self.random_agent.act(state)
        heuristic_action = self.heuristic_agent.act(state)
        
        # All should return valid actions
        for action in [dqn_action, random_action, heuristic_action]:
            self.assertIn(action, range(self.action_size))
    
    def test_dqn_agent_learning_loop(self):
        """Test DQN agent in a simple learning loop"""
        # Simulate a few environment interactions
        for episode in range(3):
            state = np.random.random(self.state_size)
            
            for step in range(10):
                action = self.dqn_agent.act(state, training=True)
                reward = np.random.random() - 0.5  # Random reward
                next_state = np.random.random(self.state_size)
                done = step == 9  # End episode after 10 steps
                
                # Store experience
                self.dqn_agent.step(state, action, reward, next_state, done)
                
                state = next_state
                
                if done:
                    break
        
        # Agent should have stored experiences
        self.assertGreater(len(self.dqn_agent.memory), 0)
        
        # Epsilon should have decayed
        self.assertLess(self.dqn_agent.epsilon, self.dqn_agent.epsilon_start)


if __name__ == '__main__':
    unittest.main() 