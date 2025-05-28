import unittest
import numpy as np
import sys
import os
import tempfile
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
        self.state_size = self.env.width * self.env.height        
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
        total_reward = 0
        steps = 0
        done = False
        while not done and steps < 50:            
            state = obs.flatten()
            action = self.dqn_agent.act(state, training=True)
            self.assertIn(action, range(self.action_size))
            next_obs, reward, done, info = self.env.step(action)
            next_state = next_obs.flatten()
            self.dqn_agent.step(state, action, reward, next_state, done)
            obs = next_obs
            total_reward += reward
            steps += 1
        self.assertGreater(steps, 0)
        self.assertIsInstance(total_reward, (int, float))
        self.assertGreater(len(self.dqn_agent.memory), 0)
    def test_random_agent_environment_loop(self):
        """Test random agent interacting with environment"""
        obs = self.env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done and steps < 50:
            action = self.random_agent.act(obs.flatten())
            self.assertIn(action, range(self.action_size))
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1
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
            while not done and steps < 30:                
                state = obs.flatten()
                action = self.dqn_agent.act(state, training=True)
                next_obs, reward, done, info = self.env.step(action)
                next_state = next_obs.flatten()
                self.dqn_agent.step(state, action, reward, next_state, done)
                obs = next_obs
                total_reward += reward
                steps += 1
            episode_rewards.append(total_reward)
        self.assertEqual(len(episode_rewards), 3)
        for reward in episode_rewards:
            self.assertIsInstance(reward, (int, float))
    def test_agent_learning_progression(self):
        """Test that DQN agent's epsilon decreases over time"""
        initial_epsilon = self.dqn_agent.epsilon
        obs = self.env.reset()
        for _ in range(20):
            state = obs.flatten()
            action = self.dqn_agent.act(state, training=True)
            next_obs, reward, done, info = self.env.step(action)
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs
        self.assertLess(self.dqn_agent.epsilon, initial_epsilon)
class TestPreprocessingIntegration(unittest.TestCase):
    """Integration tests with preprocessing"""
    def setUp(self):
        """Set up test fixtures"""
        self.env = TetrisEnv()
        self.preprocessor = TetrisPreprocessor(use_pca=False)
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
        features = self.preprocessor.extract_features(obs)
        self.assertEqual(len(features), self.preprocessor.get_feature_size())
        steps = 0
        done = False
        while not done and steps < 20:
            features = self.preprocessor.extract_features(obs)
            action = self.agent.act(features, training=True)
            next_obs, reward, done, info = self.env.step(action)
            next_features = self.preprocessor.extract_features(next_obs)
            self.agent.step(features, action, reward, next_features, done)
            obs = next_obs
            steps += 1
        self.assertGreater(steps, 0)
        self.assertGreater(len(self.agent.memory), 0)
    def test_preprocessing_with_pca(self):
        """Test preprocessing with PCA enabled"""
        preprocessor_pca = TetrisPreprocessor(use_pca=True, n_components=5)
        observations = []
        obs = self.env.reset()
        observations.append(obs)
        for _ in range(15):
            action = np.random.randint(self.env.action_space)
            obs, _, done, _ = self.env.step(action)
            observations.append(obs)
            if done:
                obs = self.env.reset()
        features_list = []
        for observation in observations:
            features = preprocessor_pca.extract_features(observation)
            features_list.append(features)
        transformed_features = preprocessor_pca.fit_transform(features_list)
        self.assertEqual(transformed_features.shape[1], 5)
        agent_pca = DQNAgent(
            state_size=5,
            action_size=self.env.action_space,
            buffer_size=50,
            batch_size=8
        )
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
        initial_observations = []
        obs = self.env.reset()
        initial_observations.append(obs)
        for _ in range(20):
            action = np.random.randint(self.env.action_space)
            obs, _, done, _ = self.env.step(action)
            initial_observations.append(obs)
            if done:
                obs = self.env.reset()
        features_list = [self.preprocessor.extract_features(obs) for obs in initial_observations]
        self.preprocessor.fit_transform(features_list)
        episode_rewards = []
        for episode in range(3):
            obs = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            while not done and steps < 30:
                features = self.preprocessor.extract_features(obs)
                processed_features = self.preprocessor.transform(features)
                action = self.agent.act(processed_features, training=True)
                next_obs, reward, done, info = self.env.step(action)
                next_features = self.preprocessor.extract_features(next_obs)
                next_processed_features = self.preprocessor.transform(next_features)
                self.agent.step(processed_features, action, reward, next_processed_features, done)
                obs = next_obs
                total_reward += reward
                steps += 1
            episode_rewards.append(total_reward)
        self.assertEqual(len(episode_rewards), 3)
        self.assertGreater(len(self.agent.memory), 0)
        self.assertLess(self.agent.epsilon, self.agent.epsilon_start)
    def test_save_load_integration(self):
        """Test saving and loading agent in integrated system"""
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
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            self.agent.save(tmp_path)
            new_agent = DQNAgent(
                state_size=self.preprocessor.get_feature_size(),
                action_size=self.env.action_space
            )
            new_agent.load(tmp_path)
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
        total_reward = 0
        steps = 0
        done = False
        while not done and steps < 20:
            features = self.preprocessor.extract_features(obs)
            action = self.agent.act(features, training=False)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1
        self.assertGreater(steps, 0)
        self.assertIsInstance(total_reward, (int, float))
    def test_different_board_states(self):
        """Test system with various board states"""
        scenarios = []
        for _ in range(5):
            obs = self.env.reset()
            for _ in range(np.random.randint(5, 15)):
                action = np.random.randint(self.env.action_space)
                obs, _, done, _ = self.env.step(action)
                if done:
                    break
            scenarios.append(obs)
        for i, board_state in enumerate(scenarios):
            with self.subTest(scenario=i):
                features = self.preprocessor.extract_features(board_state)
                self.assertEqual(len(features), self.preprocessor.get_feature_size())
                action = self.agent.act(features, training=False)
                self.assertIn(action, range(self.env.action_space))
class TestErrorHandling(unittest.TestCase):
    """Test error handling in integrated system"""
    def setUp(self):
        """Set up test fixtures"""
        self.env = TetrisEnv()
        self.agent = DQNAgent(
            state_size=220,            action_size=self.env.action_space
        )
    def test_invalid_action_handling(self):
        """Test system behavior with invalid actions"""
        obs = self.env.reset()
        for action in [-1, 4, 10, 100]:            
            try:
                next_obs, reward, done, info = self.env.step(action)
                self.assertIsInstance(next_obs, np.ndarray)
                self.assertIsInstance(reward, (int, float))
                self.assertIsInstance(done, bool)
                self.assertIsInstance(info, dict)
            except (ValueError, IndexError):
                pass
    def test_malformed_state_handling(self):
        """Test agent behavior with malformed states"""
        wrong_size_state = np.random.random(100)        
        try:
            action = self.agent.act(wrong_size_state)
            self.assertIn(action, range(self.env.action_space))
        except (RuntimeError, ValueError):
            pass
if __name__ == '__main__':
    unittest.main() 