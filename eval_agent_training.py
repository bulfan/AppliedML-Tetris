import argparse
import random
import numpy as np
import torch
from multiprocessing import Pool, cpu_count
import json
import matplotlib.pyplot as plt

from env.game_data import BOARD_DATA, BoardData, Shape
from agents.evaluation_agent import EvaluationAgent

def compute_reward(lines: int, done: bool) -> float:
    """Replicate reward logic from TetrisEnv.step."""
    reward = lines
    if done:
        reward -= 10
    
    #higher reward for multiple lines cleared
    line_reward = [
        0, 1, 3, 5, 10
    ]
    if lines < len(line_reward):
        reward += line_reward[lines]
    else:
        # no different reward for more than 4 lines cleared
        reward += 10
    
    board = np.array(BOARD_DATA.getData(), dtype=np.float32).reshape(
        BoardData.height, BoardData.width
    )

    #punishment for holes in the board (empty space with a block above)
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
    
    #punishment for height of the board
    highest_block = 0
    for x in range(BoardData.width):
        for y in range(BoardData.height):
            if board[y, x] > 0:
                highest_block = max(highest_block, BoardData.height - y)
                break
    reward -= 0.005 * highest_block

    return float(reward)


def run_episode(args) -> float:
    agent, max_steps = args
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


def train(pop_size: int, steps: int, episodes: int, elite_size: int, sigma: float, model_out: str, num_workers: int = None):
    # Initialize population of agents
    population = [EvaluationAgent() for _ in range(pop_size)]
    workers = num_workers or min(cpu_count(), pop_size)
    print(f"Using {workers} parallel workers for evaluation.")
    avg_rewards = []
    best_rewards = []

    with Pool(processes=workers) as pool:
        for ep in range(1, episodes + 1):
            args_list = [(agent, steps) for agent in population]
            rewards = pool.map(run_episode, args_list)

            scores = list(zip(rewards, population))
            scores.sort(key=lambda x: x[0], reverse=True)
            best_reward = scores[0][0]
            avg_reward = sum(rewards) / len(rewards)

            best_rewards.append(best_reward)
            avg_rewards.append(avg_reward)

            print(f"Generation {ep:3d} - Best: {best_reward:.2f} | Avg: {avg_reward:.2f}")

            # Save checkpoint every 100 episodes
            if ep % 100 == 0:
                checkpoint_path = model_out.replace(".pth", f"_ep{ep}.pth")
                torch.save(scores[0][1].state_dict(), checkpoint_path)
                print(f"average reward of 100 episodes: {sum(avg_rewards[-100:]) / 100:.2f}")
                print(f"Checkpoint saved at {checkpoint_path}")


            elites = [agent for _, agent in scores[:elite_size]]
            new_population = []
            for elite in elites:
                clone = EvaluationAgent()
                clone.load_state_dict(elite.state_dict())
                new_population.append(clone)
            
            # add 10 random agents to the new population
            for _ in range(10):
                random_agent = EvaluationAgent()
                new_population.append(random_agent)

            # Generate mutated offspring
            while len(new_population) < pop_size:
                parent = random.choice(elites)
                child = EvaluationAgent()
                child.load_state_dict(mutate_state(parent.state_dict(), sigma))
                new_population.append(child)

            population = new_population

    # Save best model from final generation
    best_agent = scores[0][1]
    torch.save(best_agent.state_dict(), model_out)
    print(f"Saved best agent's state to {model_out}")
    log_path = model_out.replace(".pth", "_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "best_rewards": best_rewards,
            "avg_rewards": avg_rewards
        }, f)
    print(f"Logged rewards to {log_path}")
    plot_train(avg_rewards, best_rewards, model_out)

def plot_train(avg_rewards: list, best_rewards: list, model_out: str):
    plt.plot(avg_rewards, label="Average Reward")
    plt.plot(best_rewards, label="Best Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Training plot saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic training for EvaluationAgent")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--elite-size", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--model-out", type=str, default="AppliedML-Tetris/models/evaluation_agent.pth")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (defaults to CPU count)")
    args = parser.parse_args()
    train(args.pop_size, args.steps, args.episodes, args.elite_size, args.sigma, args.model_out, args.workers)
 