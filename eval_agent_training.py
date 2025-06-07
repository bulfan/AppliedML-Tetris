import argparse
import random
import numpy as np
import torch

from env.game_data import BOARD_DATA, BoardData, Shape
from agents.evaluation_agent import EvaluationAgent


def compute_reward(lines: int, done: bool) -> float:
    """Replicate reward logic from TetrisEnv.step."""
    reward = lines
    if done:
        reward -= 10
    if lines > 0:
        reward += 1 * lines
    board = np.array(BOARD_DATA.getData(), dtype=np.float32).reshape(
        BoardData.height, BoardData.width
    )
    holes = 0
    for x in range(BoardData.width):
        found_filled = False
        for y in range(BoardData.height):
            if board[y, x] > 0:
                found_filled = True
            elif found_filled and board[y, x] == 0:
                holes += 1
                found_filled = False
    if holes > 0:
        reward -= 0.001 * holes
    highest_block = 0
    for x in range(BoardData.width):
        for y in range(BoardData.height):
            if board[y, x] > 0:
                highest_block = max(highest_block, BoardData.height - y)
                break
    reward -= 0.005 * highest_block
    if not done:
        reward += 0.0001
    return float(reward)


def run_episode(agent: EvaluationAgent, max_steps: int) -> float:
    BOARD_DATA.clear()
    BOARD_DATA.nextShape = Shape(random.randint(1, 7))
    BOARD_DATA.createNewPiece()
    total_reward = 0.0
    for _ in range(max_steps):
        move = agent.best_move(BOARD_DATA)
        if not move:
            break
        rot, x_target, features = move
        k = 0
        while BOARD_DATA.currentDirection != rot and k < 4:
            BOARD_DATA.rotateRight()
            k += 1
        while BOARD_DATA.currentX < x_target:
            BOARD_DATA.moveRight()
        while BOARD_DATA.currentX > x_target:
            BOARD_DATA.moveLeft()
        lines = BOARD_DATA.dropDown()
        done = BOARD_DATA.currentShape.shape == Shape.shapeNone
        reward = compute_reward(lines, done)
        agent.update(features, reward)
        total_reward += reward
        if done:
            break
    return total_reward


def mutate_state(state_dict: dict[str, torch.Tensor], sigma: float) -> dict[str, torch.Tensor]:
    """Return a mutated copy of ``state_dict``."""
    new_state = {}
    for k, v in state_dict.items():
        noise = torch.randn_like(v) * sigma
        new_state[k] = v + noise
    return new_state


def train(pop_size: int, steps: int, episodes: int, elite_size: int, sigma: float, model_out: str):
    population = [EvaluationAgent() for _ in range(pop_size)]
    for ep in range(1, episodes + 1):
        scores = []
        for agent in population:
            reward = run_episode(agent, steps)
            scores.append((reward, agent))
        scores.sort(key=lambda x: x[0], reverse=True)
        best_reward = scores[0][0]
        print(f"Generation {ep:3d} - best reward {best_reward:.2f}")
        elites = [agent for _, agent in scores[:elite_size]]
        new_population = elites.copy()
        while len(new_population) < pop_size:
            parent = random.choice(elites)
            child = EvaluationAgent()
            child.load_state_dict(mutate_state(parent.state_dict(), sigma))
            new_population.append(child)
        population = new_population
    # save the best agent from the final generation
    best_model = scores[0][1]
    best_model.save(model_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic training for EvaluationAgent")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--elite-size", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--model-out", type=str, default="AppliedML-Tetris/models/evaluation_agent.pth")
    args = parser.parse_args()
    train(args.pop_size, args.steps, args.episodes, args.elite_size, args.sigma, args.model_out)