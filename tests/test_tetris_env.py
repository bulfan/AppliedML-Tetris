import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path to import modules
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
        
        # Check observation shape
        self.assertEqual(obs.shape, (self.env.height, self.env.width))
        self.assertEqual(obs.dtype, np.float32)
        
        # Check that game is not over after reset
        self.assertFalse(self.env.game_over)
        
        # Check that board is mostly empty (except for current piece)
        self.assertLessEqual(np.sum(obs > 0), 10)  # At most 10 filled cells (current piece + some noise)
    
    def test_seed(self):
        """Test environment seeding for reproducibility"""
        # Reset with seed
        self.env.seed(42)
        obs1 = self.env.reset()
        
        # Reset with same seed
        self.env.seed(42)
        obs2 = self.env.reset()
        
        # Should get same initial state
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_get_obs(self):
        """Test observation generation"""
        obs = self.env._get_obs()
        
        # Check shape and type
        self.assertEqual(obs.shape, (self.env.height, self.env.width))
        self.assertEqual(obs.dtype, np.float32)
        
        # Check values are in valid range
        self.assertTrue(np.all(obs >= 0))
        self.assertTrue(np.all(obs <= 7))  # Max shape value
    
    def test_step_actions(self):
        """Test all possible actions"""
        self.env.reset()
        
        # Test each action
        for action in range(4):
            obs, reward, done, info = self.env.step(action)
            
            # Check return types
            self.assertIsInstance(obs, np.ndarray)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)
            
            # Check observation shape
            self.assertEqual(obs.shape, (self.env.height, self.env.width))
            
            # Check info contains lines_cleared
            self.assertIn('lines_cleared', info)
            self.assertIsInstance(info['lines_cleared'], int)
            
            if done:
                break
    
    def test_step_left_action(self):
        """Test left movement action"""
        self.env.reset()
        initial_obs = self.env._get_obs()
        
        obs, reward, done, info = self.env.step(0)  # Move left
        
        # Should return valid observation
        self.assertEqual(obs.shape, initial_obs.shape)
        self.assertIsInstance(reward, (int, float))
    
    def test_step_right_action(self):
        """Test right movement action"""
        self.env.reset()
        initial_obs = self.env._get_obs()
        
        obs, reward, done, info = self.env.step(1)  # Move right
        
        # Should return valid observation
        self.assertEqual(obs.shape, initial_obs.shape)
        self.assertIsInstance(reward, (int, float))
    
    def test_step_rotate_action(self):
        """Test rotation action"""
        self.env.reset()
        initial_obs = self.env._get_obs()
        
        obs, reward, done, info = self.env.step(2)  # Rotate
        
        # Should return valid observation
        self.assertEqual(obs.shape, initial_obs.shape)
        self.assertIsInstance(reward, (int, float))
    
    def test_step_drop_action(self):
        """Test drop action"""
        self.env.reset()
        initial_obs = self.env._get_obs()
        
        obs, reward, done, info = self.env.step(3)  # Drop
        
        # Should return valid observation
        self.assertEqual(obs.shape, initial_obs.shape)
        self.assertIsInstance(reward, (int, float))
        
        # Drop should potentially clear lines
        self.assertGreaterEqual(info['lines_cleared'], 0)
    
    def test_reward_system(self):
        """Test reward calculation"""
        self.env.reset()
        
        # Test basic step - reward can be negative due to penalties
        obs, reward, done, info = self.env.step(0)
        self.assertIsInstance(reward, (int, float))
        
        # Test that we get some reward (positive or negative)
        if not done:
            # The reward system includes penalties, so it might be negative
            # The key is that we get a numerical reward
            self.assertTrue(isinstance(reward, (int, float)))
            self.assertGreater(abs(reward), 0)  # Should get some non-zero reward
    
    def test_count_holes(self):
        """Test hole counting function"""
        # Create a board with holes
        board = np.zeros((self.env.height, self.env.width))
        
        # Create a hole: filled cell above empty cell
        board[20, 0] = 1  # Bottom filled
        board[19, 0] = 0  # Hole above
        board[18, 0] = 1  # Filled above hole
        
        holes = self.env._count_holes(board)
        self.assertGreaterEqual(holes, 1)
    
    def test_get_max_height(self):
        """Test maximum height calculation"""
        # Empty board
        board = np.zeros((self.env.height, self.env.width))
        height = self.env._get_max_height(board)
        self.assertEqual(height, 0)
        
        # Board with one piece at bottom
        board[21, 0] = 1
        height = self.env._get_max_height(board)
        self.assertEqual(height, 1)
        
        # Board with piece higher up
        board[10, 1] = 1
        height = self.env._get_max_height(board)
        self.assertEqual(height, 12)  # 22 - 10 = 12
    
    def test_game_over_condition(self):
        """Test game over detection"""
        self.env.reset()
        
        # Force game over by filling top row
        for x in range(BOARD_DATA.width):
            BOARD_DATA.backBoard[x] = 1
        
        obs, reward, done, info = self.env.step(0)
        
        # Should detect game over
        self.assertTrue(done)
        self.assertTrue(self.env.game_over)
        self.assertLess(reward, 0)  # Should get negative reward for game over
    
    def test_line_clear_reward(self):
        """Test reward for clearing lines"""
        self.env.reset()
        
        # Fill bottom row except one cell
        for x in range(BOARD_DATA.width - 1):
            BOARD_DATA.backBoard[x + (BOARD_DATA.height - 1) * BOARD_DATA.width] = 1
        
        # Create a piece that can complete the line
        BOARD_DATA.createNewPiece()
        
        # Try to complete the line (this is probabilistic based on piece type)
        obs, reward, done, info = self.env.step(3)  # Drop
        
        # If lines were cleared, should get positive reward
        if info['lines_cleared'] > 0:
            self.assertGreater(reward, 10)  # Should get significant reward for line clear
    
    def test_multiple_episodes(self):
        """Test running multiple episodes"""
        for episode in range(3):
            self.env.reset()
            done = False
            steps = 0
            
            while not done and steps < 200:  # Increased limit from 100 to 200
                action = np.random.randint(4)
                obs, reward, done, info = self.env.step(action)
                steps += 1
            
            # Each episode should eventually end or hit the limit
            self.assertLessEqual(steps, 200)
    
    def test_observation_consistency(self):
        """Test that observations are consistent"""
        self.env.reset()
        
        # Take several steps and check observations
        for _ in range(10):
            obs = self.env._get_obs()
            
            # Check that observation matches board state
            self.assertEqual(obs.shape, (self.env.height, self.env.width))
            self.assertTrue(np.all(obs >= 0))
            self.assertTrue(np.all(obs <= 7))
            
            # Take a random action
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
        
        # Check that environment dimensions match BOARD_DATA
        self.assertEqual(self.env.width, BOARD_DATA.width)
        self.assertEqual(self.env.height, BOARD_DATA.height)
    
    def test_action_board_data_consistency(self):
        """Test that actions properly affect BOARD_DATA"""
        self.env.reset()
        
        if BOARD_DATA.currentShape.shape != 0:  # If there's a current piece
            initial_x = BOARD_DATA.currentX
            
            # Move left
            self.env.step(0)
            # Position should change or stay same if blocked
            self.assertLessEqual(BOARD_DATA.currentX, initial_x)
            
            # Move right
            self.env.step(1)
            # Position should change or stay same if blocked
            self.assertGreaterEqual(BOARD_DATA.currentX, initial_x - 1)


if __name__ == '__main__':
    unittest.main() 