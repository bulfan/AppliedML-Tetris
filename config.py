# Configuration file for Tetris RL project
# All hyperparameters and paths

import os

# Environment settings
BOARD_WIDTH = 10
BOARD_HEIGHT = 22
TETROMINO_SHAPES = 7

# DQN Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQUENCY = 100

# Training settings
MAX_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000
SAVE_FREQUENCY = 100

# Network architecture
HIDDEN_LAYERS = [512, 256, 128]
DROPOUT_RATE = 0.2

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints")

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Reward settings
REWARD_LINE_CLEAR = {
    1: 40,    # Single line
    2: 100,   # Double lines
    3: 300,   # Triple lines
    4: 1200   # Tetris (4 lines)
}
REWARD_GAME_OVER = -100
REWARD_STEP = -1
REWARD_HEIGHT_PENALTY = -0.5

# Feature extraction settings
USE_PCA = False
PCA_COMPONENTS = 50
NORMALIZE_FEATURES = True 