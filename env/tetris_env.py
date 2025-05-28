import numpy as np
from env.game_data import BOARD_DATA, Shape
import random

class TetrisEnv:
    """
    A simple Gym-like wrapper around game_data.BoardData.
    Actions: 0=left, 1=right, 2=rotate, 3=drop
    """
    def __init__(self):
        self.action_space = 4  # Updated to match our 4 actions
        self.width = BOARD_DATA.width
        self.height = BOARD_DATA.height
        self.game_over = False
        self.reset()

    def reset(self):
        BOARD_DATA.clear()
        # re-seed next piece
        BOARD_DATA.nextShape = Shape(np.random.randint(1, 8))
        # spawn first piece
        BOARD_DATA.createNewPiece()
        self.game_over = False
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
        Actions: 0=left, 1=right, 2=rotate, 3=drop
        """
        done = False
        reward = 0
        lines = 0

        # Execute action
        if action == 0:  # Move left
            BOARD_DATA.moveLeft()
        elif action == 1:  # Move right
            BOARD_DATA.moveRight()
        elif action == 2:  # Rotate
            BOARD_DATA.rotateRight()
        elif action == 3:  # Drop
            lines = BOARD_DATA.dropDown()
        
        # Always move down after action (except for drop)
        if action != 3:
            lines = BOARD_DATA.moveDown()

        # Check for game over
        if BOARD_DATA.gameOver():
            done = True
            self.game_over = True
            reward -= 10

        # Reward for line clears
        if lines > 0:
            reward += lines * 10
            print(f"Cleared {lines} lines, reward: {lines * 10}")
        
        # Small reward for staying alive
        if not done:
            reward += 0.1

        # Calculate additional rewards based on board state
        board = self._get_obs()
        
        # Penalty for holes
        holes = self._count_holes(board)
        if holes > 0:
            reward -= holes * 0.5
        
        # Penalty for height
        max_height = self._get_max_height(board)
        reward -= max_height * 0.1

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