"""
Quick Training Script for Tetris AI Models
This script quickly trains multiple AI models for demonstration purposes.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from env.tetris_env import TetrisEnv
from agents.ai_manager import AI_MANAGER
import config
def quick_train_agent(agent_name, episodes=100):
    """Quickly train an agent for demonstration"""
    print(f"\n Quick training {agent_name} for {episodes} episodes...")
    if not AI_MANAGER.set_agent(agent_name):
        print(f" Failed to switch to agent {agent_name}")
        return
    if not agent_name.startswith("dqn"):
        print(f"  {agent_name} is not a trainable agent")
        return
    env = TetrisEnv()
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 500        
        while not env.game_over and steps < max_steps:
            action = AI_MANAGER.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            AI_MANAGER.train_step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
        total_rewards.append(total_reward)
        if episode % 20 == 0:
            avg_reward = np.mean(total_rewards[-20:]) if len(total_rewards) >= 20 else np.mean(total_rewards)
            agent_info = AI_MANAGER.get_agent_info()
            epsilon = agent_info.get('epsilon', 0.0)
            print(f"  Episode {episode:3d}: Avg Reward: {avg_reward:6.1f}, Epsilon: {epsilon:.3f}")
    AI_MANAGER.save_current_model()
    final_avg = np.mean(total_rewards[-10:]) if len(total_rewards) >= 10 else np.mean(total_rewards)
    print(f" Training completed! Final average reward: {final_avg:.1f}")
    return total_rewards
def main():
    """Main training function"""
    print(" Quick Training Script for Tetris AI Models")
    print("=" * 60)
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    basic_rewards = quick_train_agent("dqn_basic", episodes=50)
    advanced_rewards = quick_train_agent("dqn_advanced", episodes=100)
    print("\n Training Summary:")
    print("=" * 60)
    if basic_rewards:
        print(f"DQN Basic - Final Avg Reward: {np.mean(basic_rewards[-10:]):.1f}")
    if advanced_rewards:
        print(f"DQN Advanced - Final Avg Reward: {np.mean(advanced_rewards[-10:]):.1f}")
    print("\n Quick training completed!")
    print(" You can now run the game and switch between different AI agents!")
    print(" Use the API to test the trained models!")
if __name__ == "__main__":
    main() 