import unittest
import os
import sys

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config


class TestConfig(unittest.TestCase):
    """Test cases for the config module"""
    
    def test_board_dimensions(self):
        """Test board dimension constants"""
        self.assertEqual(config.BOARD_WIDTH, 10)
        self.assertEqual(config.BOARD_HEIGHT, 22)
        self.assertIsInstance(config.BOARD_WIDTH, int)
        self.assertIsInstance(config.BOARD_HEIGHT, int)
        self.assertGreater(config.BOARD_WIDTH, 0)
        self.assertGreater(config.BOARD_HEIGHT, 0)
    
    def test_tetromino_shapes(self):
        """Test tetromino shape constant"""
        self.assertEqual(config.TETROMINO_SHAPES, 7)
        self.assertIsInstance(config.TETROMINO_SHAPES, int)
        self.assertGreater(config.TETROMINO_SHAPES, 0)
    
    def test_learning_parameters(self):
        """Test learning-related parameters"""
        # Learning rate
        self.assertEqual(config.LEARNING_RATE, 0.001)
        self.assertIsInstance(config.LEARNING_RATE, float)
        self.assertGreater(config.LEARNING_RATE, 0)
        self.assertLess(config.LEARNING_RATE, 1)
        
        # Gamma (discount factor)
        self.assertEqual(config.GAMMA, 0.99)
        self.assertIsInstance(config.GAMMA, float)
        self.assertGreater(config.GAMMA, 0)
        self.assertLessEqual(config.GAMMA, 1)
        
        # Epsilon parameters
        self.assertEqual(config.EPSILON_START, 1.0)
        self.assertEqual(config.EPSILON_END, 0.01)
        self.assertEqual(config.EPSILON_DECAY, 0.995)
        
        self.assertIsInstance(config.EPSILON_START, float)
        self.assertIsInstance(config.EPSILON_END, float)
        self.assertIsInstance(config.EPSILON_DECAY, float)
        
        self.assertGreaterEqual(config.EPSILON_START, config.EPSILON_END)
        self.assertGreater(config.EPSILON_DECAY, 0)
        self.assertLess(config.EPSILON_DECAY, 1)
    
    def test_training_parameters(self):
        """Test training-related parameters"""
        # Batch size
        self.assertEqual(config.BATCH_SIZE, 32)
        self.assertIsInstance(config.BATCH_SIZE, int)
        self.assertGreater(config.BATCH_SIZE, 0)
        
        # Memory size
        self.assertEqual(config.MEMORY_SIZE, 10000)
        self.assertIsInstance(config.MEMORY_SIZE, int)
        self.assertGreater(config.MEMORY_SIZE, config.BATCH_SIZE)
        
        # Target update frequency
        self.assertEqual(config.TARGET_UPDATE_FREQUENCY, 100)
        self.assertIsInstance(config.TARGET_UPDATE_FREQUENCY, int)
        self.assertGreater(config.TARGET_UPDATE_FREQUENCY, 0)
    
    def test_episode_parameters(self):
        """Test episode-related parameters"""
        # Max episodes
        self.assertEqual(config.MAX_EPISODES, 1000)
        self.assertIsInstance(config.MAX_EPISODES, int)
        self.assertGreater(config.MAX_EPISODES, 0)
        
        # Max steps per episode
        self.assertEqual(config.MAX_STEPS_PER_EPISODE, 1000)
        self.assertIsInstance(config.MAX_STEPS_PER_EPISODE, int)
        self.assertGreater(config.MAX_STEPS_PER_EPISODE, 0)
        
        # Save frequency
        self.assertEqual(config.SAVE_FREQUENCY, 100)
        self.assertIsInstance(config.SAVE_FREQUENCY, int)
        self.assertGreater(config.SAVE_FREQUENCY, 0)
    
    def test_network_architecture(self):
        """Test neural network architecture parameters"""
        # Hidden layers
        self.assertEqual(config.HIDDEN_LAYERS, [512, 256, 128])
        self.assertIsInstance(config.HIDDEN_LAYERS, list)
        self.assertGreater(len(config.HIDDEN_LAYERS), 0)
        
        for layer_size in config.HIDDEN_LAYERS:
            self.assertIsInstance(layer_size, int)
            self.assertGreater(layer_size, 0)
        
        # Dropout rate
        self.assertEqual(config.DROPOUT_RATE, 0.2)
        self.assertIsInstance(config.DROPOUT_RATE, float)
        self.assertGreaterEqual(config.DROPOUT_RATE, 0)
        self.assertLess(config.DROPOUT_RATE, 1)
    
    def test_path_configurations(self):
        """Test path-related configurations"""
        # Project root should exist
        self.assertTrue(os.path.exists(config.PROJECT_ROOT))
        self.assertTrue(os.path.isdir(config.PROJECT_ROOT))
        
        # Model save path
        self.assertIsInstance(config.MODEL_SAVE_PATH, str)
        self.assertTrue(os.path.exists(config.MODEL_SAVE_PATH))
        self.assertTrue(os.path.isdir(config.MODEL_SAVE_PATH))
        
        # Log path
        self.assertIsInstance(config.LOG_PATH, str)
        self.assertTrue(os.path.exists(config.LOG_PATH))
        self.assertTrue(os.path.isdir(config.LOG_PATH))
        
        # Checkpoint path
        self.assertIsInstance(config.CHECKPOINT_PATH, str)
        self.assertTrue(os.path.exists(config.CHECKPOINT_PATH))
        self.assertTrue(os.path.isdir(config.CHECKPOINT_PATH))
    
    def test_reward_configurations(self):
        """Test reward-related configurations"""
        # Reward line clear dictionary
        self.assertIsInstance(config.REWARD_LINE_CLEAR, dict)
        self.assertIn(1, config.REWARD_LINE_CLEAR)
        self.assertIn(2, config.REWARD_LINE_CLEAR)
        self.assertIn(3, config.REWARD_LINE_CLEAR)
        self.assertIn(4, config.REWARD_LINE_CLEAR)
        
        # Check reward values are reasonable
        self.assertEqual(config.REWARD_LINE_CLEAR[1], 40)
        self.assertEqual(config.REWARD_LINE_CLEAR[2], 100)
        self.assertEqual(config.REWARD_LINE_CLEAR[3], 300)
        self.assertEqual(config.REWARD_LINE_CLEAR[4], 1200)
        
        # Rewards should increase with more lines
        rewards = [config.REWARD_LINE_CLEAR[i] for i in range(1, 5)]
        self.assertEqual(rewards, sorted(rewards))
        
        # Game over reward
        self.assertEqual(config.REWARD_GAME_OVER, -100)
        self.assertIsInstance(config.REWARD_GAME_OVER, int)
        self.assertLess(config.REWARD_GAME_OVER, 0)
        
        # Step reward
        self.assertEqual(config.REWARD_STEP, -1)
        self.assertIsInstance(config.REWARD_STEP, int)
        self.assertLess(config.REWARD_STEP, 0)
        
        # Height penalty
        self.assertEqual(config.REWARD_HEIGHT_PENALTY, -0.5)
        self.assertIsInstance(config.REWARD_HEIGHT_PENALTY, float)
        self.assertLess(config.REWARD_HEIGHT_PENALTY, 0)
    
    def test_preprocessing_configurations(self):
        """Test preprocessing-related configurations"""
        # PCA usage
        self.assertEqual(config.USE_PCA, False)
        self.assertIsInstance(config.USE_PCA, bool)
        
        # PCA components
        self.assertEqual(config.PCA_COMPONENTS, 50)
        self.assertIsInstance(config.PCA_COMPONENTS, int)
        self.assertGreater(config.PCA_COMPONENTS, 0)
        
        # Feature normalization
        self.assertEqual(config.NORMALIZE_FEATURES, True)
        self.assertIsInstance(config.NORMALIZE_FEATURES, bool)
    
    def test_parameter_relationships(self):
        """Test relationships between parameters"""
        # Memory size should be larger than batch size
        self.assertGreater(config.MEMORY_SIZE, config.BATCH_SIZE)
        
        # Epsilon start should be greater than epsilon end
        self.assertGreater(config.EPSILON_START, config.EPSILON_END)
        
        # Save frequency should be reasonable relative to max episodes
        self.assertLessEqual(config.SAVE_FREQUENCY, config.MAX_EPISODES)
        
        # Target update frequency should be reasonable
        self.assertLess(config.TARGET_UPDATE_FREQUENCY, config.MAX_STEPS_PER_EPISODE)
    
    def test_path_construction(self):
        """Test that paths are constructed correctly"""
        # Check that paths are within project root
        self.assertTrue(config.MODEL_SAVE_PATH.startswith(config.PROJECT_ROOT))
        self.assertTrue(config.LOG_PATH.startswith(config.PROJECT_ROOT))
        self.assertTrue(config.CHECKPOINT_PATH.startswith(config.PROJECT_ROOT))
        
        # Check path separators are correct for the OS
        expected_model_path = os.path.join(config.PROJECT_ROOT, "models")
        expected_log_path = os.path.join(config.PROJECT_ROOT, "logs")
        expected_checkpoint_path = os.path.join(config.PROJECT_ROOT, "checkpoints")
        
        self.assertEqual(config.MODEL_SAVE_PATH, expected_model_path)
        self.assertEqual(config.LOG_PATH, expected_log_path)
        self.assertEqual(config.CHECKPOINT_PATH, expected_checkpoint_path)
    
    def test_numeric_types(self):
        """Test that numeric parameters have correct types"""
        # Integer parameters
        int_params = [
            'BOARD_WIDTH', 'BOARD_HEIGHT', 'TETROMINO_SHAPES',
            'BATCH_SIZE', 'MEMORY_SIZE', 'TARGET_UPDATE_FREQUENCY',
            'MAX_EPISODES', 'MAX_STEPS_PER_EPISODE', 'SAVE_FREQUENCY',
            'PCA_COMPONENTS', 'REWARD_GAME_OVER', 'REWARD_STEP'
        ]
        
        for param in int_params:
            value = getattr(config, param)
            self.assertIsInstance(value, int, f"{param} should be int, got {type(value)}")
        
        # Float parameters
        float_params = [
            'LEARNING_RATE', 'GAMMA', 'EPSILON_START', 'EPSILON_END',
            'EPSILON_DECAY', 'DROPOUT_RATE', 'REWARD_HEIGHT_PENALTY'
        ]
        
        for param in float_params:
            value = getattr(config, param)
            self.assertIsInstance(value, float, f"{param} should be float, got {type(value)}")
        
        # Boolean parameters
        bool_params = ['USE_PCA', 'NORMALIZE_FEATURES']
        
        for param in bool_params:
            value = getattr(config, param)
            self.assertIsInstance(value, bool, f"{param} should be bool, got {type(value)}")
    
    def test_reward_line_clear_completeness(self):
        """Test that reward line clear dictionary is complete"""
        # Should have rewards for 1-4 lines (standard Tetris)
        for lines in range(1, 5):
            self.assertIn(lines, config.REWARD_LINE_CLEAR)
            self.assertIsInstance(config.REWARD_LINE_CLEAR[lines], int)
            self.assertGreater(config.REWARD_LINE_CLEAR[lines], 0)
    
    def test_hidden_layers_validity(self):
        """Test that hidden layers configuration is valid"""
        self.assertIsInstance(config.HIDDEN_LAYERS, list)
        self.assertGreater(len(config.HIDDEN_LAYERS), 0)
        
        # Each layer should have positive size
        for i, layer_size in enumerate(config.HIDDEN_LAYERS):
            self.assertIsInstance(layer_size, int, f"Layer {i} size should be int")
            self.assertGreater(layer_size, 0, f"Layer {i} size should be positive")
        
        # Layers should generally decrease in size (common practice)
        for i in range(len(config.HIDDEN_LAYERS) - 1):
            self.assertGreaterEqual(
                config.HIDDEN_LAYERS[i], 
                config.HIDDEN_LAYERS[i + 1],
                "Hidden layers should generally decrease in size"
            )


class TestConfigConstants(unittest.TestCase):
    """Test that config constants are not accidentally modified"""
    
    def test_board_constants_immutable(self):
        """Test that board constants maintain expected values"""
        # These should never change as they define the standard Tetris board
        self.assertEqual(config.BOARD_WIDTH, 10)
        self.assertEqual(config.BOARD_HEIGHT, 22)
        self.assertEqual(config.TETROMINO_SHAPES, 7)
    
    def test_directory_creation(self):
        """Test that required directories are created"""
        required_dirs = [
            config.MODEL_SAVE_PATH,
            config.LOG_PATH,
            config.CHECKPOINT_PATH
        ]
        
        for directory in required_dirs:
            self.assertTrue(os.path.exists(directory), f"Directory {directory} should exist")
            self.assertTrue(os.path.isdir(directory), f"{directory} should be a directory")


if __name__ == '__main__':
    unittest.main() 