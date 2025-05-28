import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.preprocessing import TetrisPreprocessor
import config


class TestTetrisPreprocessor(unittest.TestCase):
    """Test cases for the TetrisPreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = TetrisPreprocessor(use_pca=False, n_components=10)
        
        # Create sample board states
        self.empty_board = np.zeros((config.BOARD_HEIGHT, config.BOARD_WIDTH))
        
        # Board with some pieces
        self.sample_board = np.zeros((config.BOARD_HEIGHT, config.BOARD_WIDTH))
        self.sample_board[20:22, 0:3] = 1  # Bottom left filled
        self.sample_board[19:22, 5:7] = 2  # Bottom middle filled
        
        # Board with holes
        self.board_with_holes = np.zeros((config.BOARD_HEIGHT, config.BOARD_WIDTH))
        self.board_with_holes[21, :] = 1  # Bottom row filled
        self.board_with_holes[20, 1] = 0  # Hole in second row
        self.board_with_holes[19, 1] = 1  # Block above hole
    
    def test_initialization(self):
        """Test preprocessor initialization"""
        # Test without PCA
        preprocessor_no_pca = TetrisPreprocessor(use_pca=False)
        self.assertFalse(preprocessor_no_pca.use_pca)
        self.assertIsNone(preprocessor_no_pca.pca)
        
        # Test with PCA
        preprocessor_with_pca = TetrisPreprocessor(use_pca=True, n_components=20)
        self.assertTrue(preprocessor_with_pca.use_pca)
        self.assertIsNotNone(preprocessor_with_pca.pca)
        self.assertEqual(preprocessor_with_pca.n_components, 20)
    
    def test_extract_features_empty_board(self):
        """Test feature extraction from empty board"""
        features = self.preprocessor.extract_features(self.empty_board)
        
        # Check feature vector is numpy array
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.float32)
        
        # Check feature vector length
        expected_length = self.preprocessor.get_feature_size()
        self.assertEqual(len(features), expected_length)
        
        # For empty board, most features should be zero
        heights = features[:config.BOARD_WIDTH]  # First 10 features are heights
        self.assertTrue(np.all(heights == 0))
        
        # Max height should be 0
        max_height_idx = config.BOARD_WIDTH + (config.BOARD_WIDTH - 1) + 0
        self.assertEqual(features[max_height_idx], 0)
        
        # Holes should be 0
        holes_idx = config.BOARD_WIDTH + (config.BOARD_WIDTH - 1) + 3
        self.assertEqual(features[holes_idx], 0)
    
    def test_extract_features_sample_board(self):
        """Test feature extraction from board with pieces"""
        features = self.preprocessor.extract_features(self.sample_board)
        
        # Check feature vector properties
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.float32)
        
        # Heights should be non-zero for some columns
        heights = features[:config.BOARD_WIDTH]
        self.assertGreater(np.sum(heights > 0), 0)
        
        # Max height should be positive
        max_height_idx = config.BOARD_WIDTH + (config.BOARD_WIDTH - 1) + 0
        self.assertGreater(features[max_height_idx], 0)
        
        # Average height should be positive
        avg_height_idx = config.BOARD_WIDTH + (config.BOARD_WIDTH - 1) + 1
        self.assertGreater(features[avg_height_idx], 0)
    
    def test_get_column_heights(self):
        """Test column height calculation"""
        heights = self.preprocessor._get_column_heights(self.sample_board)
        
        self.assertEqual(len(heights), config.BOARD_WIDTH)
        
        # Check specific heights based on sample board
        # Columns 0-2 should have height 2 (filled from row 20-21)
        self.assertEqual(heights[0], 2)
        self.assertEqual(heights[1], 2)
        self.assertEqual(heights[2], 2)
        
        # Columns 5-6 should have height 3 (filled from row 19-21)
        self.assertEqual(heights[5], 3)
        self.assertEqual(heights[6], 3)
        
        # Other columns should be empty
        self.assertEqual(heights[3], 0)
        self.assertEqual(heights[4], 0)
        self.assertEqual(heights[7], 0)
    
    def test_count_holes(self):
        """Test hole counting"""
        # Empty board should have no holes
        holes_empty = self.preprocessor._count_holes(self.empty_board)
        self.assertEqual(holes_empty, 0)
        
        # Sample board should have no holes
        holes_sample = self.preprocessor._count_holes(self.sample_board)
        self.assertEqual(holes_sample, 0)
        
        # Board with holes should detect holes
        holes_with_holes = self.preprocessor._count_holes(self.board_with_holes)
        self.assertGreater(holes_with_holes, 0)
    
    def test_count_complete_lines(self):
        """Test complete line counting"""
        # Empty board should have no complete lines
        complete_empty = self.preprocessor._count_complete_lines(self.empty_board)
        self.assertEqual(complete_empty, 0)
        
        # Create board with complete line
        complete_line_board = np.zeros((config.BOARD_HEIGHT, config.BOARD_WIDTH))
        complete_line_board[21, :] = 1  # Fill bottom row completely
        
        complete_lines = self.preprocessor._count_complete_lines(complete_line_board)
        self.assertEqual(complete_lines, 1)
        
        # Create board with multiple complete lines
        complete_line_board[20, :] = 1  # Fill second-to-bottom row
        complete_lines_multi = self.preprocessor._count_complete_lines(complete_line_board)
        self.assertEqual(complete_lines_multi, 2)
    
    def test_count_wells(self):
        """Test well counting"""
        # Empty board should have no wells
        wells_empty = self.preprocessor._count_wells(self.empty_board)
        self.assertEqual(wells_empty, 0)
        
        # Create board with a well
        well_board = np.zeros((config.BOARD_HEIGHT, config.BOARD_WIDTH))
        well_board[20:22, 0] = 1  # Left wall
        well_board[20:22, 2] = 1  # Right wall
        # Column 1 is empty between walls - creates a well
        
        wells = self.preprocessor._count_wells(well_board)
        self.assertGreater(wells, 0)
    
    def test_count_blocked_cells(self):
        """Test blocked cell counting"""
        # Empty board should have no blocked cells
        blocked_empty = self.preprocessor._count_blocked_cells(self.empty_board)
        self.assertEqual(blocked_empty, 0)
        
        # Create board with blocked cells
        blocked_board = np.zeros((config.BOARD_HEIGHT, config.BOARD_WIDTH))
        blocked_board[21, 0] = 1  # Bottom cell
        blocked_board[19, 0] = 1  # Cell with empty space below
        # Row 20, column 0 is empty - creates blocked cell above
        
        blocked = self.preprocessor._count_blocked_cells(blocked_board)
        self.assertGreater(blocked, 0)
    
    def test_get_feature_size(self):
        """Test feature size calculation"""
        # Without PCA
        preprocessor_no_pca = TetrisPreprocessor(use_pca=False)
        size_no_pca = preprocessor_no_pca.get_feature_size()
        
        # Expected: heights(10) + height_diffs(9) + aggregates(3) + holes(1) + bumpiness(1) + complete_lines(1) + wells(1) + blocked_cells(1)
        expected_size = 10 + 9 + 3 + 1 + 1 + 1 + 1 + 1
        self.assertEqual(size_no_pca, expected_size)
        
        # With PCA
        preprocessor_with_pca = TetrisPreprocessor(use_pca=True, n_components=15)
        size_with_pca = preprocessor_with_pca.get_feature_size()
        self.assertEqual(size_with_pca, 15)
    
    def test_fit_transform(self):
        """Test fitting and transforming features"""
        # Create multiple feature vectors
        features_list = []
        for _ in range(10):
            # Create random board states
            board = np.random.randint(0, 2, size=(config.BOARD_HEIGHT, config.BOARD_WIDTH))
            features = self.preprocessor.extract_features(board)
            features_list.append(features)
        
        # Fit and transform
        transformed_features = self.preprocessor.fit_transform(features_list)
        
        # Check output shape
        self.assertEqual(transformed_features.shape[0], len(features_list))
        self.assertEqual(transformed_features.shape[1], self.preprocessor.get_feature_size())
        
        # Check that preprocessor is fitted
        self.assertTrue(self.preprocessor.fitted)
    
    def test_transform_after_fit(self):
        """Test transforming single feature vector after fitting"""
        # First fit the preprocessor
        features_list = []
        for _ in range(5):
            board = np.random.randint(0, 2, size=(config.BOARD_HEIGHT, config.BOARD_WIDTH))
            features = self.preprocessor.extract_features(board)
            features_list.append(features)
        
        self.preprocessor.fit_transform(features_list)
        
        # Now transform a single feature vector
        single_features = self.preprocessor.extract_features(self.sample_board)
        transformed_single = self.preprocessor.transform(single_features)
        
        # Check output shape
        self.assertEqual(len(transformed_single), self.preprocessor.get_feature_size())
        self.assertIsInstance(transformed_single, np.ndarray)
    
    def test_transform_before_fit_raises_error(self):
        """Test that transform raises error before fitting"""
        features = self.preprocessor.extract_features(self.sample_board)
        
        with self.assertRaises(ValueError):
            self.preprocessor.transform(features)
    
    def test_with_pca(self):
        """Test preprocessor with PCA enabled"""
        preprocessor_pca = TetrisPreprocessor(use_pca=True, n_components=10)
        
        # Create feature data
        features_list = []
        for _ in range(20):
            board = np.random.randint(0, 2, size=(config.BOARD_HEIGHT, config.BOARD_WIDTH))
            features = preprocessor_pca.extract_features(board)
            features_list.append(features)
        
        # Fit and transform
        transformed = preprocessor_pca.fit_transform(features_list)
        
        # Check that dimensionality is reduced
        self.assertEqual(transformed.shape[1], 10)
        self.assertTrue(preprocessor_pca.fitted)
        
        # Test single transform
        single_features = preprocessor_pca.extract_features(self.sample_board)
        transformed_single = preprocessor_pca.transform(single_features)
        self.assertEqual(len(transformed_single), 10)
    
    def test_with_normalization(self):
        """Test preprocessor with feature normalization"""
        # Create preprocessor with normalization
        preprocessor_norm = TetrisPreprocessor(use_pca=False)
        
        # Only test if scaler is actually enabled
        if preprocessor_norm.scaler is None:
            self.skipTest("Normalization is disabled in config")
        
        # Create feature data with different scales
        features_list = []
        for i in range(20):  # Increased sample size for better statistics
            board = np.random.randint(0, 2, size=(config.BOARD_HEIGHT, config.BOARD_WIDTH))
            # Add some variation to create different scales
            if i < 15:  # Fill different amounts for different samples
                board[20:22, :min(i+1, config.BOARD_WIDTH)] = 1
            features = preprocessor_norm.extract_features(board)
            features_list.append(features)
        
        # Check that we have variation in the original features
        original_features = np.array(features_list)
        original_stds = np.std(original_features, axis=0)
        
        # Only test normalization if there's actually variation to normalize
        if np.any(original_stds > 1e-6):
            # Fit and transform
            transformed = preprocessor_norm.fit_transform(features_list)
            
            # Check that features are normalized
            means = np.mean(transformed, axis=0)
            stds = np.std(transformed, axis=0)
            
            # For features that had variation, check normalization
            varied_features = original_stds > 1e-6
            if np.any(varied_features):
                # Means should be close to zero for varied features
                self.assertTrue(np.allclose(means[varied_features], 0, atol=1e-6))
                # Standard deviations should be close to 1 for varied features
                self.assertTrue(np.allclose(stds[varied_features], 1, atol=1e-6))
        else:
            self.skipTest("No variation in features to test normalization")
    
    def test_feature_consistency(self):
        """Test that feature extraction is consistent"""
        # Extract features multiple times from same board
        features1 = self.preprocessor.extract_features(self.sample_board)
        features2 = self.preprocessor.extract_features(self.sample_board)
        
        # Should be identical
        np.testing.assert_array_equal(features1, features2)
    
    def test_different_board_sizes(self):
        """Test with different board configurations"""
        # This test assumes the board size is configurable
        # If not, it will test with the default size
        
        # Test with current board size
        features = self.preprocessor.extract_features(self.sample_board)
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)


class TestTetrisPreprocessorEdgeCases(unittest.TestCase):
    """Test edge cases for TetrisPreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = TetrisPreprocessor(use_pca=False)
    
    def test_full_board(self):
        """Test with completely filled board"""
        full_board = np.ones((config.BOARD_HEIGHT, config.BOARD_WIDTH))
        features = self.preprocessor.extract_features(full_board)
        
        # All heights should be maximum
        heights = features[:config.BOARD_WIDTH]
        self.assertTrue(np.all(heights == config.BOARD_HEIGHT))
        
        # Should have no holes (since board is completely filled)
        holes_idx = config.BOARD_WIDTH + (config.BOARD_WIDTH - 1) + 3
        self.assertEqual(features[holes_idx], 0)
        
        # All lines should be complete
        complete_lines_idx = config.BOARD_WIDTH + (config.BOARD_WIDTH - 1) + 5
        self.assertEqual(features[complete_lines_idx], config.BOARD_HEIGHT)
    
    def test_single_column_filled(self):
        """Test with only one column filled"""
        single_col_board = np.zeros((config.BOARD_HEIGHT, config.BOARD_WIDTH))
        single_col_board[:, 0] = 1  # Fill first column completely
        
        features = self.preprocessor.extract_features(single_col_board)
        
        # First column height should be maximum
        self.assertEqual(features[0], config.BOARD_HEIGHT)
        
        # Other column heights should be zero
        heights = features[:config.BOARD_WIDTH]
        self.assertTrue(np.all(heights[1:] == 0))
        
        # Should have high bumpiness due to height difference
        bumpiness_idx = config.BOARD_WIDTH + (config.BOARD_WIDTH - 1) + 4
        self.assertGreater(features[bumpiness_idx], 0)
    
    def test_checkerboard_pattern(self):
        """Test with checkerboard pattern"""
        checkerboard = np.zeros((config.BOARD_HEIGHT, config.BOARD_WIDTH))
        for i in range(config.BOARD_HEIGHT):
            for j in range(config.BOARD_WIDTH):
                if (i + j) % 2 == 0:
                    checkerboard[i, j] = 1
        
        features = self.preprocessor.extract_features(checkerboard)
        
        # Should detect many holes
        holes_idx = config.BOARD_WIDTH + (config.BOARD_WIDTH - 1) + 3
        self.assertGreater(features[holes_idx], 0)
        
        # Should have no complete lines
        complete_lines_idx = config.BOARD_WIDTH + (config.BOARD_WIDTH - 1) + 5
        self.assertEqual(features[complete_lines_idx], 0)


if __name__ == '__main__':
    unittest.main() 