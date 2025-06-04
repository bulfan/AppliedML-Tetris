import os
BOARD_WIDTH = 10
BOARD_HEIGHT = 22
TETROMINO_SHAPES = 7

LEARNING_RATE = 0.001
GAMMA = 0.99

EPSILON_START = 1.0
EPSILON_END = 0.01
# EPSILON_DECAY = 0.995 decay too fast
EPSILON_DECAY = 976  # good for 5000 episodes when decay starts after 500 episodes
# ─ OR ─
# Option B: If you'd rather do a simple linear schedule, leave EPSILON_DECAY unused 
#            and implement inside your loop: 
#       epsilon = max(EPSILON_END, EPSILON_START − episode*(EPSILON_START − EPSILON_END)/MAX_EPISODES)

BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQUENCY = 100

MAX_EPISODES = 5000   # train for 5000 episodes instead of 1000
MAX_STEPS_PER_EPISODE = 1000

SAVE_FREQUENCY = 100

HIDDEN_LAYERS = [512, 256, 128]
DROPOUT_RATE = 0.2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

REWARD_LINE_CLEAR = {
    1: 1000, 2: 3000, 3: 8000, 4: 20000  # 1: 40,    2: 100,    3: 300,    4: 1200   
}
REWARD_GAME_OVER = -3000
REWARD_HOLES_PENALTY = -0.35
REWARD_STEP = -0.50
REWARD_HEIGHT_PENALTY = -0.50  # -0.5

USE_PCA = False
PCA_COMPONENTS = 50
NORMALIZE_FEATURES = True