import unittest
import numpy as np
import sys
import os
import tempfile

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from env.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent, RandomAgent
from utils.preprocessing import TetrisPreprocessor
import config


class TestEnvironmentAgentIntegration(unittest.TestCase):
    """Integration tests between environment and agents"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.env = TetrisEnv()
        self.state_size = self.env.width * self.env.height  # Flattened board
        self.action_size = self.env.action_space
        
        self.dqn_agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            buffer_size=100,
            batch_size=8
        )
        self.random_agent = RandomAgent(self.action_size)
    
    def test_dqn_agent_environment_loop(self):
        """Test DQN agent interacting with environment"""
        obs = self.env.reset()
        self.assertIsNotNone(obs)
        
        # Run a short episode
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 50:  # Limit steps
            # Flatten observation for agent
            state = obs.flatten()
            
            # Agent selects action
            action = self.dqn_agent.act(state, training=True)
            self.assertIn(action, range(self.action_size))
            
            # Environment step
            next_obs, reward, done, info = self.env.step(action)
            
            # Store experience
            next_state = next_obs.flatten()
            self.dqn_agent.step(state, action, reward, next_state, done)
            
            # Update for next iteration
            obs = next_obs
            total_reward += reward
            steps += 1
        
        # Check that episode completed
        self.assertGreater(steps, 0)
        self.assertIsInstance(total_reward, (int, float))
        
        # Check that agent stored experiences
        self.assertGreater(len(self.dqn_agent.memory), 0)
    
    def test_random_agent_environment_loop(self):
        """Test random agent interacting with environment"""
        obs = self.env.reset()
        
        # Run a short episode
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 50:
            # Agent selects random action
            action = self.random_agent.act(obs.flatten())
            self.assertIn(action, range(self.action_size))
            
            # Environment step
            obs, reward, done, info = self.env.step(action)
            
            total_reward += reward
            steps += 1
        
        # Check that episode completed
        self.assertGreater(steps, 0)
        self.assertIsInstance(total_reward, (int, float))
    
    def test_multiple_episodes(self):
        """Test running multiple episodes with DQN agent"""
        episode_rewards = []
        
        for episode in range(3):
            obs = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 30:  # Short episodes
                state = obs.flatten()
                action = self.dqn_agent.act(state, training=True)
                next_obs, reward, done, info = self.env.step(action)
                
                next_state = next_obs.flatten()
                self.dqn_agent.step(state, action, reward, next_state, done)
                
                obs = next_obs
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
        
        # Check that all episodes completed
        self.assertEqual(len(episode_rewards), 3)
        for reward in episode_rewards:
            self.assertIsInstance(reward, (int, float))
    
    def test_agent_learning_progression(self):
        """Test that DQN agent's epsilon decreases over time"""
        initial_epsilon = self.dqn_agent.epsilon
        
        # Run several steps to trigger epsilon decay
        obs = self.env.reset()
        for _ in range(20):
            state = obs.flatten()
            action = self.dqn_agent.act(state, training=True)
            next_obs, reward, done, info = self.env.step(action)
            
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs
        
        # Epsilon should have decreased
        self.assertLess(self.dqn_agent.epsilon, initial_epsilon)


class TestPreprocessingIntegration(unittest.TestCase):
    """Integration tests with preprocessing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.env = TetrisEnv()
        self.preprocessor = TetrisPreprocessor(use_pca=False)
        
        # Create agent with preprocessed features
        feature_size = self.preprocessor.get_feature_size()
        self.agent = DQNAgent(
            state_size=feature_size,
            action_size=self.env.action_space,
            buffer_size=50,
            batch_size=8
        )
    
    def test_environment_preprocessing_agent_loop(self):
        """Test full loop with preprocessing"""
        obs = self.env.reset()
        
        # Extract features from initial observation
        features = self.preprocessor.extract_features(obs)
        self.assertEqual(len(features), self.preprocessor.get_feature_size())
        
        # Run episode with feature extraction
        steps = 0
        done = False
        
        while not done and steps < 20:
            # Extract features
            features = self.preprocessor.extract_features(obs)
            
            # Agent acts on features
            action = self.agent.act(features, training=True)
            
            # Environment step
            next_obs, reward, done, info = self.env.step(action)
            
            # Extract features from next state
            next_features = self.preprocessor.extract_features(next_obs)
            
            # Store experience with features
            self.agent.step(features, action, reward, next_features, done)
            
            obs = next_obs
            steps += 1
        
        # Check that episode completed successfully
        self.assertGreater(steps, 0)
        self.assertGreater(len(self.agent.memory), 0)
    
    def test_preprocessing_with_pca(self):
        """Test preprocessing with PCA enabled"""
        preprocessor_pca = TetrisPreprocessor(use_pca=True, n_components=5)
        
        # Collect some observations to fit PCA
        observations = []
        obs = self.env.reset()
        observations.append(obs)
        
        for _ in range(15):
            action = np.random.randint(self.env.action_space)
            obs, _, done, _ = self.env.step(action)
            observations.append(obs)
            
            if done:
                obs = self.env.reset()
        
        # Extract features and fit PCA
        features_list = []
        for observation in observations:
            features = preprocessor_pca.extract_features(observation)
            features_list.append(features)
        
        # Fit PCA
        transformed_features = preprocessor_pca.fit_transform(features_list)
        
        # Check dimensionality reduction
        self.assertEqual(transformed_features.shape[1], 5)
        
        # Test with agent
        agent_pca = DQNAgent(
            state_size=5,
            action_size=self.env.action_space,
            buffer_size=50,
            batch_size=8
        )
        
        # Test single step
        obs = self.env.reset()
        features = preprocessor_pca.extract_features(obs)
        transformed_features = preprocessor_pca.transform(features)
        
        action = agent_pca.act(transformed_features, training=True)
        self.assertIn(action, range(self.env.action_space))


class TestFullSystemIntegration(unittest.TestCase):
    """Full system integration tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.env = TetrisEnv()
        self.preprocessor = TetrisPreprocessor(use_pca=False)
        
        feature_size = self.preprocessor.get_feature_size()
        self.agent = DQNAgent(
            state_size=feature_size,
            action_size=self.env.action_space,
            buffer_size=100,
            batch_size=16
        )
    
    def test_training_simulation(self):
        """Simulate a short training session"""
        # Collect initial data for preprocessing
        initial_observations = []
        obs = self.env.reset()
        initial_observations.append(obs)
        
        # Collect some random data
        for _ in range(20):
            action = np.random.randint(self.env.action_space)
            obs, _, done, _ = self.env.step(action)
            initial_observations.append(obs)
            
            if done:
                obs = self.env.reset()
        
        # Fit preprocessor
        features_list = [self.preprocessor.extract_features(obs) for obs in initial_observations]
        self.preprocessor.fit_transform(features_list)
        
        # Training loop
        episode_rewards = []
        
        for episode in range(3):
            obs = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 30:
                # Preprocess observation
                features = self.preprocessor.extract_features(obs)
                processed_features = self.preprocessor.transform(features)
                
                # Agent action
                action = self.agent.act(processed_features, training=True)
                
                # Environment step
                next_obs, reward, done, info = self.env.step(action)
                
                # Preprocess next observation
                next_features = self.preprocessor.extract_features(next_obs)
                next_processed_features = self.preprocessor.transform(next_features)
                
                # Store experience
                self.agent.step(processed_features, action, reward, next_processed_features, done)
                
                obs = next_obs
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
        
        # Verify training completed
        self.assertEqual(len(episode_rewards), 3)
        self.assertGreater(len(self.agent.memory), 0)
        
        # Check that agent learned (epsilon decreased)
        self.assertLess(self.agent.epsilon, self.agent.epsilon_start)
    
    def test_save_load_integration(self):
        """Test saving and loading agent in integrated system"""
        # Train agent briefly
        obs = self.env.reset()
        for _ in range(10):
            features = self.preprocessor.extract_features(obs)
            action = self.agent.act(features, training=True)
            next_obs, reward, done, info = self.env.step(action)
            
            next_features = self.preprocessor.extract_features(next_obs)
            self.agent.step(features, action, reward, next_features, done)
            
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs
        
        # Save agent
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            self.agent.save(tmp_path)
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_size=self.preprocessor.get_feature_size(),
                action_size=self.env.action_space
            )
            new_agent.load(tmp_path)
            
            # Test that loaded agent works
            obs = self.env.reset()
            features = self.preprocessor.extract_features(obs)
            action = new_agent.act(features, training=False)
            
            self.assertIn(action, range(self.env.action_space))
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_evaluation_mode(self):
        """Test system in evaluation mode (no training)"""
        obs = self.env.reset()
        
        # Run episode in evaluation mode
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 20:
            features = self.preprocessor.extract_features(obs)
            
            # Agent in evaluation mode (no training)
            action = self.agent.act(features, training=False)
            
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1
        
        # Should complete without errors
        self.assertGreater(steps, 0)
        self.assertIsInstance(total_reward, (int, float))
    
    def test_different_board_states(self):
        """Test system with various board states"""
        # Test with different scenarios
        scenarios = []
        
        # Collect diverse board states
        for _ in range(5):
            obs = self.env.reset()
            
            # Take some random actions to create different states
            for _ in range(np.random.randint(5, 15)):
                action = np.random.randint(self.env.action_space)
                obs, _, done, _ = self.env.step(action)
                
                if done:
                    break
            
            scenarios.append(obs)
        
        # Test preprocessing and agent on each scenario
        for i, board_state in enumerate(scenarios):
            with self.subTest(scenario=i):
                # Extract features
                features = self.preprocessor.extract_features(board_state)
                self.assertEqual(len(features), self.preprocessor.get_feature_size())
                
                # Agent should handle all scenarios
                action = self.agent.act(features, training=False)
                self.assertIn(action, range(self.env.action_space))


class TestErrorHandling(unittest.TestCase):
    """Test error handling in integrated system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.env = TetrisEnv()
        self.agent = DQNAgent(
            state_size=220,  # Flattened board
            action_size=self.env.action_space
        )
    
    def test_invalid_action_handling(self):
        """Test system behavior with invalid actions"""
        obs = self.env.reset()
        
        # Test that environment handles actions gracefully
        # (Environment should clamp or ignore invalid actions)
        for action in [-1, 4, 10, 100]:  # Invalid actions
            try:
                next_obs, reward, done, info = self.env.step(action)
                # If no exception, check that we got valid outputs
                self.assertIsInstance(next_obs, np.ndarray)
                self.assertIsInstance(reward, (int, float))
                self.assertIsInstance(done, bool)
                self.assertIsInstance(info, dict)
            except (ValueError, IndexError):
                # It's also acceptable for environment to raise errors
                pass
    
    def test_malformed_state_handling(self):
        """Test agent behavior with malformed states"""
        # Test with wrong-sized state
        wrong_size_state = np.random.random(100)  # Wrong size
        
        try:
            action = self.agent.act(wrong_size_state)
            # If no exception, action should still be valid
            self.assertIn(action, range(self.env.action_space))
        except (RuntimeError, ValueError):
            # It's acceptable for agent to raise errors with wrong input size
            pass


if __name__ == '__main__':
    unittest.main() 