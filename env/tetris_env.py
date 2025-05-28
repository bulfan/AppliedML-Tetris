import numpy as np
from game_data import BOARD_DATA, Shape
import random

class TetrisEnv:
    """
    A simple Gym-like wrapper around game_data.BoardData.
    Actions: 0=noop(soft-drop), 1=left, 2=right, 3=rotate, 4=drop
    """
    def __init__(self):
        self.action_space = 5
        self.width = BOARD_DATA.width
        self.height = BOARD_DATA.height
        self.reset()

    def reset(self):
        BOARD_DATA.clear()
        # re-seed next piece
        BOARD_DATA.nextShape = Shape(np.random.randint(1, 8))
        # spawn first piece
        BOARD_DATA.createNewPiece()
        return self._get_obs()
    
    def seed(self, seed: int):
        """Seed the RNGs so that reset() is repeatable."""
        np.random.seed(seed)
        random.seed(seed)

    def _get_obs(self):
        # overlay current piece onto board copy
        board = np.array(BOARD_DATA.getData(), dtype=np.float32).reshape(self.height, self.width)
        # draw current shape
        for x, y in BOARD_DATA.currentShape.getCoords(BOARD_DATA.currentDirection,
                                                     BOARD_DATA.currentX,
                                                     BOARD_DATA.currentY):
            if 0 <= x < self.width and 0 <= y < self.height:
                board[y, x] = BOARD_DATA.currentShape.shape
        return board  # shape=(height, width)

    def step(self, action):
        """
        Perform the action, return obs, reward, done.
        reward = #lines cleared, -10 on game over.
        """
        done = False
        reward = 0

        if action == 0:
            lines = BOARD_DATA.moveDown()
        elif action == 1:
            BOARD_DATA.moveLeft()
            lines = BOARD_DATA.moveDown()
        elif action == 2:
            BOARD_DATA.moveRight()
            lines = BOARD_DATA.moveDown()
        elif action == 3:
            BOARD_DATA.rotateRight()
            lines = BOARD_DATA.moveDown()
        elif action == 4:
            lines = BOARD_DATA.dropDown()
        else:
            lines = 0

        reward += lines
        # if after the move the current piece is “none”, we hit game over
        if BOARD_DATA.currentShape.shape == Shape.shapeNone:
            done = True
            reward -= 10

        # reward for line clears
        if lines > 0:
            reward += 100 * lines
            print(f"Cleared {lines} lines, total reward: {reward}")

        return self._get_obs(), reward, done, {}