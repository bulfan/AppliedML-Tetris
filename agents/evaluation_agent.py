from typing import Mapping
import torch
import torch.nn as nn
import copy
from env.game_data import BOARD_DATA, BoardData
import numpy as np
from typing import Any
import gzip
from pathlib import Path

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

    def __init__(self, hidden_size: int = 20, lr: float = 5e-5,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self._plan: list[int] = []  # Stores the current plan of actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(next(self.parameters()).device)
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

        # punishement for holes in the board (empty space or spaces with a block above)
        holes = 0
        for x in range(BoardData.width):
            found_filled = False
            for y in range(BoardData.height):
                if board.backBoard[x + y * BoardData.width] > 0:
                    found_filled = True
                elif found_filled and board.backBoard[x + y * BoardData.width] == 0:
                    holes += 1
                    found_filled = False
        
        # max and min height of columns
        max_h = max(heights) if heights else 0
        min_h = min(heights) if heights else 0

        #multiple lines cleared in one move gives extra reward
        line_reward = [
            0,  1,  3,  5, 10
        ]
        if lines_cleared < len(line_reward):
            lines_cleared = line_reward[lines_cleared]
        else:
        # no different reward for more than 4 lines cleared
            lines_cleared = 10
        
        # Normalize and invert features where necessary
        norm_diff_height = -diff_height / (BoardData.height * BoardData.width)
        norm_holes = -holes / (BoardData.height * BoardData.width)
        norm_max_h = -max_h / BoardData.height
        norm_min_h = -min_h / BoardData.height
        norm_lines = lines_cleared

        feats = torch.tensor([
            norm_diff_height,
            norm_holes,
            norm_max_h,
            norm_min_h,
            norm_lines
        ], dtype=torch.float32)
        return feats

    # ------------------------------------------------------------------
    # Move simulation
    # ------------------------------------------------------------------
    def _simulate_move(self, rotation: int, target_x: int, board: BoardData) -> tuple[float, torch.Tensor, int, int] | None:
        """Simulate dropping the current piece with ``rotation`` and ``target_x``.

        Returns ``(predicted_reward, features, rotation, target_x)`` or ``None``
        if the move is impossible.
        """
        board = copy.deepcopy(board)
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

    def best_move(self, board: BoardData) -> tuple[int, int, torch.Tensor] | None:
        """Evaluate all moves and return the best one based on predicted reward."""
        best = None
        best_reward = -float('inf')
        for rot in range(4):
            for x in range(BoardData.width):
                res = self._simulate_move(rot, x, board)
                if res is None:
                    continue
                reward, feats, r_used, x_used = res
                if reward > best_reward:
                    best_reward = reward
                    best = (r_used, x_used, feats)
        return best
    
    def act(self, board: BoardData) -> list[int] | None:
        """Return the best move based on current board state."""
        move = self.best_move(board)
        if move is None:
            return None
        rot, x_target, features = move
        k = 0
        list_of_moves = []
        # make sure not to use the real board, but a copy
        board = copy.deepcopy(board)
        while board.currentDirection != rot and k < 4:
            board.rotateRight()
            k += 1
            list_of_moves.append(2)
        while board.currentX < x_target:
            board.moveRight()
            list_of_moves.append(1)
        while board.currentX > x_target:
            board.moveLeft()
            list_of_moves.append(0)
        list_of_moves.append(3)
        self.update(features, 0)
        return list_of_moves

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update(self, features, actual_reward):
        self.optimizer.zero_grad()
        pred = self.forward(features)
        target = torch.tensor([actual_reward], dtype=torch.float32, device=pred.device)
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()
    
    def save(self, filepath: str):
        """Save the model state to the specified file."""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load the model state from the specified file."""
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        self.eval()



    # ------------------------------------------------------------------
    # TorchScript export for standalone loading
    # ------------------------------------------------------------------
    def export_script(self) -> torch.jit.ScriptModule:
        """Return a TorchScript version of the model for standalone use."""
        # Script only the network module
        scripted = torch.jit.script(self.net)
        return scripted

    def save_scripted(self, filepath: str, compress: bool = True, compresslevel: int = 9):
        """Convenience: export to TorchScript and save in one step."""
        scripted = self.export_script()
        EvaluationAgent.save_script(scripted, filepath, compress=compress, compresslevel=compresslevel)

    @staticmethod
    def save_script(scripted_module: torch.jit.ScriptModule, filepath: str, compress: bool = True, compresslevel: int = 9):
        """Save a TorchScript module to disk, optionally gzipped."""
        parent = Path(filepath).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        if compress:
            # Save to a temporary file then gzip-compress
            tmp_path = filepath + ".tmp"
            scripted_module.save(tmp_path)
            with gzip.open(filepath, 'wb', compresslevel=compresslevel) as f_out:
                with open(tmp_path, 'rb') as f_in:
                    f_out.write(f_in.read())
            # remove temporary file
            Path(tmp_path).unlink()
        else:
            scripted_module.save(filepath)

    @staticmethod
    def load_script(filepath: str, map_location: Any = 'cpu', compress: bool = True) -> torch.jit.ScriptModule:
        """Load a TorchScript module (gzipped or raw) for inference without class code."""
        if compress:
            with gzip.open(filepath, 'rb') as f:
                scripted = torch.jit.load(f, map_location=map_location)
        else:
            scripted = torch.jit.load(filepath, map_location=map_location)
        return scripted

    
