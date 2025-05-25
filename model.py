import numpy as np
import tensorflow as tf
from game_data import BoardData


def build_model(input_shape, num_actions):
    """
    Build and compile a simple CNN mapping Tetris observations to action probabilities.
    Observations are 2-channel grids: static blocks + current piece mask.
    Actions: 0=noop, 1=move left, 2=move right, 3=rotate right, 4=rotate left, 5=hard drop.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


class TetrisAI:
    """
    AI agent that uses a CNN model to choose actions for the Tetris UI.
    """
    def __init__(self, model):
        self.model = model

    def nextMove(self, board_data):
        # Build observation: static board + current piece mask
        data = board_data.getData()  # flat list
        grid = np.array(data, dtype=np.float32).reshape(BoardData.height, BoardData.width)
        mask = np.zeros_like(grid)
        for x, y in board_data.getCurrentShapeCoord():
            if 0 <= x < BoardData.width and 0 <= y < BoardData.height:
                mask[y, x] = 1.0
        obs = np.stack([grid, mask], axis=-1)[None, ...]
        preds = self.model.predict(obs, verbose=0)[0]
        return int(np.argmax(preds))


# Instantiate the model and AI for use by game_UI
input_shape = (BoardData.height, BoardData.width, 2)
num_actions = 6
model = build_model(input_shape, num_actions)
TETRIS_AI = TetrisAI(model)

if __name__ == '__main__':
    # Quick test: play a game with the untrained model
    from game_data import BOARD_DATA
    total_lines = 0
    BOARD_DATA.clear()
    BOARD_DATA.createNewPiece()
    done = False
    while not done:
        action = TETRIS_AI.nextMove(BOARD_DATA)
        # Apply action
        if action == 1:
            BOARD_DATA.moveLeft()
        elif action == 2:
            BOARD_DATA.moveRight()
        elif action == 3:
            BOARD_DATA.rotateRight()
        elif action == 4:
            BOARD_DATA.rotateLeft()
        elif action == 5:
            BOARD_DATA.dropDown()
        # Gravity
        lines = 0 if action == 5 else BOARD_DATA.moveDown()
        total_lines += lines
        done = BOARD_DATA.gameOver()
    print(f"Test game over, lines cleared: {total_lines}")