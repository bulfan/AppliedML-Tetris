import numpy as np
from env.game_data import BOARD_DATA, Shape
import random
import config


class TetrisEnv:
    """
    A simple Gym-like wrapper around game_data.BoardData.
    Actions: 0=left, 1=right, 2=rotate, 3=drop
    """
    def __init__(self):
        self.action_space = 4        
        self.width = BOARD_DATA.width
        self.height = BOARD_DATA.height
        self.game_over = False
        self.reset()

    def reset(self):
        BOARD_DATA.clear()
        BOARD_DATA.nextShape = Shape(np.random.randint(1, 8))
        BOARD_DATA.createNewPiece()
        self.game_over = False
        return self._get_obs()
    
    def seed(self, seed: int):
        """Seed the RNGs so that reset() is repeatable."""
        np.random.seed(seed)
        random.seed(seed)

    def _get_obs(self):
        board = np.array(BOARD_DATA.getData(), dtype=np.float32).reshape(self.height, self.width)
        for x, y in BOARD_DATA.currentShape.getCoords(BOARD_DATA.currentDirection,
                                                     BOARD_DATA.currentX,
                                                     BOARD_DATA.currentY):
            if 0 <= x < self.width and 0 <= y < self.height:
                board[y, x] = BOARD_DATA.currentShape.shape
        return board
    
    def step(self, action):
        """
        Perform the action, return obs, reward, done.
        Actions: 0=left, 1=right, 2=rotate, 3=drop
        """
        done = False
        reward = 0
        lines = 0
        if action == 0:            
            BOARD_DATA.moveLeft()
        elif action == 1:            
            BOARD_DATA.moveRight()
        elif action == 2:            
            BOARD_DATA.rotateRight()
        elif action == 3:            
            lines = BOARD_DATA.dropDown()
        if action != 3:
            lines = BOARD_DATA.moveDown()
        if BOARD_DATA.gameOver():
            done = True
            self.game_over = True
            reward += config.REWARD_GAME_OVER
        if lines > 0:
            if lines <= 4:
                reward += config.REWARD_LINE_CLEAR[int(lines)]
            else:
                reward += (lines - 4) * 20000
            print(f"Cleared {lines} lines, reward: {config.REWARD_LINE_CLEAR[int(lines)] if lines <= 4 else (lines - 4) * 3000}")
        if not done:
            reward += config.REWARD_STEP
        board = self._get_obs()
        holes = self._count_holes(board)
        if holes > 0:
            reward += holes * config.REWARD_HOLES_PENALTY
        max_height = self._get_max_height(board)
        reward += max_height * config.REWARD_HEIGHT_PENALTY
        return self._get_obs(), reward, done, {'lines_cleared': lines}
    
    def _count_holes(self, board):
        """Count holes in the board"""
        holes = 0
        for x in range(self.width):
            found_filled = False
            for y in range(self.height):
                if board[y, x] > 0:
                    found_filled = True
                elif found_filled and board[y, x] == 0:
                    holes += 1
        return holes
    
    def _get_max_height(self, board):
        """Get the maximum height of pieces on the board"""
        max_height = 0
        for x in range(self.width):
            for y in range(self.height):
                if board[y, x] > 0:
                    max_height = max(max_height, self.height - y)
                    break
        return max_height