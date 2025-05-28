import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env.tetris_env import TetrisEnv
from env.game_data import BOARD_DATA
class TestTetrisEnv(unittest.TestCase):
    """Test cases for the TetrisEnv class"""
    def setUp(self):
        """Set up test fixtures"""
        self.env = TetrisEnv()
    def test_initialization(self):
        """Test environment initialization"""
        self.assertEqual(self.env.action_space, 4)
        self.assertEqual(self.env.width, 10)
        self.assertEqual(self.env.height, 22)
        self.assertFalse(self.env.game_over)
    def test_reset(self):
        """Test environment reset"""
        obs = self.env.reset()
        self.assertEqual(obs.shape, (self.env.height, self.env.width))
        self.assertEqual(obs.dtype, np.float32)
        self.assertFalse(self.env.game_over)
        self.assertLessEqual(np.sum(obs > 0), 10)    
    def test_seed(self):
        """Test environment seeding for reproducibility"""
        self.env.seed(42)
        obs1 = self.env.reset()
        self.env.seed(42)
        obs2 = self.env.reset()
        np.testing.assert_array_equal(obs1, obs2)
    def test_get_obs(self):
        """Test observation generation"""
        obs = self.env._get_obs()
        self.assertEqual(obs.shape, (self.env.height, self.env.width))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(obs >= 0))
        self.assertTrue(np.all(obs <= 7))    
    def test_step_actions(self):
        """Test all possible actions"""
        self.env.reset()
        for action in range(4):
            obs, reward, done, info = self.env.step(action)
            self.assertIsInstance(obs, np.ndarray)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)
            self.assertEqual(obs.shape, (self.env.height, self.env.width))
            self.assertIn('lines_cleared', info)
            self.assertIsInstance(info['lines_cleared'], int)
            if done:
                break
    def test_step_left_action(self):
        """Test left movement action"""
        self.env.reset()
        initial_obs = self.env._get_obs()
        obs, reward, done, info = self.env.step(0)        
        self.assertEqual(obs.shape, initial_obs.shape)
        self.assertIsInstance(reward, (int, float))
    def test_step_right_action(self):
        """Test right movement action"""
        self.env.reset()
        initial_obs = self.env._get_obs()
        obs, reward, done, info = self.env.step(1)        
        self.assertEqual(obs.shape, initial_obs.shape)
        self.assertIsInstance(reward, (int, float))
    def test_step_rotate_action(self):
        """Test rotation action"""
        self.env.reset()
        initial_obs = self.env._get_obs()
        obs, reward, done, info = self.env.step(2)        
        self.assertEqual(obs.shape, initial_obs.shape)
        self.assertIsInstance(reward, (int, float))
    def test_step_drop_action(self):
        """Test drop action"""
        self.env.reset()
        initial_obs = self.env._get_obs()
        obs, reward, done, info = self.env.step(3)        
        self.assertEqual(obs.shape, initial_obs.shape)
        self.assertIsInstance(reward, (int, float))
        self.assertGreaterEqual(info['lines_cleared'], 0)
    def test_reward_system(self):
        """Test reward calculation"""
        self.env.reset()
        obs, reward, done, info = self.env.step(0)
        self.assertIsInstance(reward, (int, float))
        if not done:
            self.assertTrue(isinstance(reward, (int, float)))
            self.assertGreater(abs(reward), 0)    
    def test_count_holes(self):
        """Test hole counting function"""
        board = np.zeros((self.env.height, self.env.width))
        board[20, 0] = 1        
        board[19, 0] = 0        
        board[18, 0] = 1        
        holes = self.env._count_holes(board)
        self.assertGreaterEqual(holes, 1)
    def test_get_max_height(self):
        """Test maximum height calculation"""
        board = np.zeros((self.env.height, self.env.width))
        height = self.env._get_max_height(board)
        self.assertEqual(height, 0)
        board[21, 0] = 1
        height = self.env._get_max_height(board)
        self.assertEqual(height, 1)
        board[10, 1] = 1
        height = self.env._get_max_height(board)
        self.assertEqual(height, 12)    
    def test_game_over_condition(self):
        """Test game over detection"""
        self.env.reset()
        for x in range(BOARD_DATA.width):
            BOARD_DATA.backBoard[x] = 1
        obs, reward, done, info = self.env.step(0)
        self.assertTrue(done)
        self.assertTrue(self.env.game_over)
        self.assertLess(reward, 0)    
    def test_line_clear_reward(self):
        """Test reward for clearing lines"""
        self.env.reset()
        for x in range(BOARD_DATA.width - 1):
            BOARD_DATA.backBoard[x + (BOARD_DATA.height - 1) * BOARD_DATA.width] = 1
        BOARD_DATA.createNewPiece()
        obs, reward, done, info = self.env.step(3)        
        if info['lines_cleared'] > 0:
            self.assertGreater(reward, 10)    
    def test_multiple_episodes(self):
        """Test running multiple episodes"""
        for episode in range(3):
            self.env.reset()
            done = False
            steps = 0
            while not done and steps < 200:                
                action = np.random.randint(4)
                obs, reward, done, info = self.env.step(action)
                steps += 1
            self.assertLessEqual(steps, 200)
    def test_observation_consistency(self):
        """Test that observations are consistent"""
        self.env.reset()
        for _ in range(10):
            obs = self.env._get_obs()
            self.assertEqual(obs.shape, (self.env.height, self.env.width))
            self.assertTrue(np.all(obs >= 0))
            self.assertTrue(np.all(obs <= 7))
            action = np.random.randint(4)
            obs, reward, done, info = self.env.step(action)
            if done:
                break
class TestTetrisEnvIntegration(unittest.TestCase):
    """Integration tests for TetrisEnv with game_data"""
    def setUp(self):
        """Set up test fixtures"""
        self.env = TetrisEnv()
    def test_env_board_data_integration(self):
        """Test that environment properly integrates with BOARD_DATA"""
        self.env.reset()
        self.assertEqual(self.env.width, BOARD_DATA.width)
        self.assertEqual(self.env.height, BOARD_DATA.height)
    def test_action_board_data_consistency(self):
        """Test that actions properly affect BOARD_DATA"""
        self.env.reset()
        if BOARD_DATA.currentShape.shape != 0:            
            initial_x = BOARD_DATA.currentX
            self.env.step(0)
            self.assertLessEqual(BOARD_DATA.currentX, initial_x)
            self.env.step(1)
            self.assertGreaterEqual(BOARD_DATA.currentX, initial_x - 1)
if __name__ == '__main__':
    unittest.main() 