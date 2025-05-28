import torch
import numpy as np
from env.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent
from torch.utils.tensorboard import SummaryWriter
from agents.replay_buffer import ReplayBuffer
summary_writer = SummaryWriter("runs/tetris_dqn")

def train(num_episodes=1000, max_steps=10000):
    env = TetrisEnv()
    agent = DQNAgent(env)
    num_episodes  = 1000
    eval_interval = 50
    eval_seeds    = [42 + i for i in range(20)]

    for ep in range(1, num_episodes+1):
        state = env.reset()
        total_reward = 0
        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            agent.optimize()
            state = next_state
            total_reward += reward
            if done:
                break

        print(f"Episode {ep:3d} – steps {t:4d}, reward {total_reward}")
        if ep % 50 == 0:
            torch.save(agent.policy_net.state_dict(), f"checkpoint_{ep}.pth")
        
        if ep % eval_interval == 0:
            eval_reward = 0
            for seed in eval_seeds:
                env.seed(seed)
                state = env.reset()
                done = False
                while not done:
                    action = agent.select_action(state)
                    state, reward, done, _ = env.step(action)
                    eval_reward += reward
            eval_reward /= len(eval_seeds)
            print(f"Evaluation after {ep} episodes: Avg Reward {eval_reward:.2f}")
            summary_writer.add_scalar('Eval/Reward', eval_reward, ep)

if __name__ == "__main__":
    train()
