# Feature preprocessing utilities for Tetris RL
# Includes feature scaling, PCA, and state representation

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import config

class TetrisPreprocessor:
    """Handles feature extraction and preprocessing for Tetris states"""
    
    def __init__(self, use_pca=config.USE_PCA, n_components=config.PCA_COMPONENTS):
        self.use_pca = use_pca
        self.n_components = n_components
        self.scaler = StandardScaler() if config.NORMALIZE_FEATURES else None
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.fitted = False
    
    def extract_features(self, board_state, current_piece=None):
        """Extract features from the current board state"""
        features = []
        
        # Board height features
        heights = self._get_column_heights(board_state)
        features.extend(heights)
        
        # Height differences between adjacent columns
        height_diffs = [heights[i+1] - heights[i] for i in range(len(heights)-1)]
        features.extend(height_diffs)
        
        # Aggregate height features
        features.append(np.max(heights))  # Max height
        features.append(np.mean(heights))  # Average height
        features.append(np.std(heights))   # Height variance
        
        # Hole features
        holes = self._count_holes(board_state)
        features.append(holes)
        
        # Bumpiness (sum of absolute height differences)
        bumpiness = sum(abs(diff) for diff in height_diffs)
        features.append(bumpiness)
        
        # Complete lines
        complete_lines = self._count_complete_lines(board_state)
        features.append(complete_lines)
        
        # Wells (deep single-column gaps)
        wells = self._count_wells(board_state)
        features.append(wells)
        
        # Blocked cells (cells with empty cells above them)
        blocked_cells = self._count_blocked_cells(board_state)
        features.append(blocked_cells)
        
        return np.array(features, dtype=np.float32)
    
    def _get_column_heights(self, board):
        """Get the height of each column"""
        heights = []
        for col in range(config.BOARD_WIDTH):
            height = 0
            for row in range(config.BOARD_HEIGHT):
                if board[row][col] != 0:
                    height = config.BOARD_HEIGHT - row
                    break
            heights.append(height)
        return heights
    
    def _count_holes(self, board):
        """Count holes in the board (empty cells with filled cells above)"""
        holes = 0
        for col in range(config.BOARD_WIDTH):
            found_block = False
            for row in range(config.BOARD_HEIGHT):
                if board[row][col] != 0:
                    found_block = True
                elif found_block and board[row][col] == 0:
                    holes += 1
        return holes
    
    def _count_complete_lines(self, board):
        """Count complete lines that can be cleared"""
        complete_lines = 0
        for row in range(config.BOARD_HEIGHT):
            if all(board[row][col] != 0 for col in range(config.BOARD_WIDTH)):
                complete_lines += 1
        return complete_lines
    
    def _count_wells(self, board):
        """Count wells (single-column deep gaps)"""
        wells = 0
        heights = self._get_column_heights(board)
        
        for col in range(config.BOARD_WIDTH):
            left_height = heights[col-1] if col > 0 else config.BOARD_HEIGHT
            right_height = heights[col+1] if col < config.BOARD_WIDTH-1 else config.BOARD_HEIGHT
            
            if heights[col] < left_height and heights[col] < right_height:
                wells += min(left_height, right_height) - heights[col]
        
        return wells
    
    def _count_blocked_cells(self, board):
        """Count cells that are blocked by cells above them"""
        blocked = 0
        for col in range(config.BOARD_WIDTH):
            for row in range(config.BOARD_HEIGHT-1, -1, -1):
                if board[row][col] != 0:
                    # Check if there are empty cells above this filled cell
                    for above_row in range(row):
                        if board[above_row][col] == 0:
                            blocked += 1
                            break
        return blocked
    
    def fit_transform(self, features_list):
        """Fit the preprocessor and transform features"""
        features_array = np.array(features_list)
        
        if self.scaler:
            features_array = self.scaler.fit_transform(features_array)
        
        if self.pca:
            features_array = self.pca.fit_transform(features_array)
        
        self.fitted = True
        return features_array
    
    def transform(self, features):
        """Transform features using fitted preprocessor"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        features_array = np.array(features).reshape(1, -1)
        
        if self.scaler:
            features_array = self.scaler.transform(features_array)
        
        if self.pca:
            features_array = self.pca.transform(features_array)
        
        return features_array.flatten()
    
    def get_feature_size(self):
        """Get the size of the feature vector after preprocessing"""
        if self.use_pca:
            return self.n_components
        else:
            # Base features: heights(10) + height_diffs(9) + aggregates(3) + holes(1) + bumpiness(1) + complete_lines(1) + wells(1) + blocked_cells(1)
            return config.BOARD_WIDTH + (config.BOARD_WIDTH-1) + 3 + 1 + 1 + 1 + 1 + 1 