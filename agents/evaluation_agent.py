from typing import Mapping
import torch
import torch.nn as nn
import copy
from env.game_data import BOARD_DATA, BoardData
import numpy as np
from typing import Any

class EvaluationAgent(nn.Module):
    """Simple evaluation agent using board heuristics.

    The network has 5 input features corresponding to:
        1. Total difference in height of adjacent columns
        2. Number of holes
        3. Maximum column height
        4. Minimum column height
        5. Lines cleared by the move
    It outputs a single value representing the predicted reward for a move.
    """

    def __init__(self, hidden_size: int = 16, lr: float = 1e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self._plan: list[int] = []  # Stores the current plan of actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ------------------------------------------------------------------
    # Heuristic calculations
    # ------------------------------------------------------------------
    @staticmethod
    def _column_height(board: BoardData, x: int) -> int:
        """Return the height (from bottom) of column ``x`` on ``board``."""
        for y in range(BoardData.height):
            if board.backBoard[x + y * BoardData.width] > 0:
                return BoardData.height - y
        return 0

    def _board_features(self, board: BoardData, lines_cleared: int) -> torch.Tensor:
        """Compute heuristic feature vector for ``board``."""
        heights = [self._column_height(board, x) for x in range(BoardData.width)]
        diff_height = sum(abs(heights[i] - heights[i - 1])
                          for i in range(1, BoardData.width))

        holes = 0
        for x in range(BoardData.width):
            block_found = False
            for y in range(BoardData.height):
                val = board.backBoard[x + y * BoardData.width]
                if val > 0:
                    block_found = True
                elif block_found and val == 0:
                    holes += 1
        max_h = max(heights) if heights else 0
        min_h = min(heights) if heights else 0

        feats = torch.tensor([
            float(diff_height),
            float(holes),
            float(max_h),
            float(min_h),
            float(lines_cleared)
        ], dtype=torch.float32)
        return feats

    # ------------------------------------------------------------------
    # Move simulation
    # ------------------------------------------------------------------
    def _simulate_move(self, rotation: int, target_x: int):
        """Simulate dropping the current piece with ``rotation`` and ``target_x``.

        Returns ``(predicted_reward, features, rotation, target_x)`` or ``None``
        if the move is impossible.
        """
        board = copy.deepcopy(BOARD_DATA)

        # rotate piece
        attempts = 0
        while board.currentDirection != rotation and attempts < 4:
            prev = board.currentDirection
            board.rotateRight()
            if board.currentDirection == prev:
                break
            attempts += 1
        if board.currentDirection != rotation:
            return None

        # move horizontally
        while board.currentX < target_x:
            prev = board.currentX
            board.moveRight()
            if board.currentX == prev:
                return None
        while board.currentX > target_x:
            prev = board.currentX
            board.moveLeft()
            if board.currentX == prev:
                return None

        lines = board.dropDown()
        feats = self._board_features(board, lines)
        with torch.no_grad():
            reward = self.forward(feats).item()
        return reward, feats, rotation, target_x

    def best_move(self):
        """Evaluate all moves and return the best one based on predicted reward."""
        best = None
        best_reward = -float('inf')
        for rot in range(4):
            for x in range(BoardData.width):
                res = self._simulate_move(rot, x)
                if res is None:
                    continue
                reward, feats, r_used, x_used = res
                if reward > best_reward:
                    best_reward = reward
                    best = (r_used, x_used, feats)
        return best
    
    def act(self):
        """Return the best move based on current board state."""
        move = self.best_move()
        if move is None:
            return None
        rot, x_target, features = move
        k = 0
        list_of_moves = []
        while BOARD_DATA.currentDirection != rot and k < 4:
            list_of_moves.append(2)
            k += 1
        while BOARD_DATA.currentX < x_target:
            list_of_moves.append(1)
        while BOARD_DATA.currentX > x_target:
            list_of_moves.append(0)
        list_of_moves.append(3)
        self.update(features, 0)
        return list_of_moves

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update(self, features: torch.Tensor, actual_reward: float) -> float:
        """Update the network given observed ``actual_reward`` for ``features``."""
        pred = self.forward(features)
        loss = self.loss_fn(pred, torch.tensor([actual_reward], dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def save(self, filepath: str):
        """Save the model state to the specified file."""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load the model state from the specified file."""
        print(f"Loading model from {filepath}")
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)

    
