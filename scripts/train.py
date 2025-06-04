"""
Comprehensive Training Script for Tetris Deep Q-Network
This script provides proper training for Tetris AI agents with appropriate
epochs, learning schedules, and evaluation metrics.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from env.tetris_env import TetrisEnv
from agents.ai_manager import AI_MANAGER
import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TetrisTrainer:
    """Comprehensive trainer for Tetris AI agents"""
    def __init__(self, agent_name="dqn_advanced"):
        self.agent_name = agent_name
        self.env = TetrisEnv()
        self.total_episodes = 2000        
        self.eval_frequency = 100        
        self.save_frequency = 200        
        self.max_steps_per_episode = 2000        
        self.training_rewards = []
        self.training_lines = []
        self.training_steps = []
        self.eval_scores = []
        self.eval_episodes = []
        self.best_score = float('-inf')
        self.best_model_path = None
        print(f" Initializing Tetris Trainer for {agent_name}")
        print(f" Training Episodes: {self.total_episodes}")
        print(f" Max Steps per Episode: {self.max_steps_per_episode}")

    def train_agent(self):
        """Main training loop"""
        print(f"\n Starting training for {self.agent_name}")
        print("=" * 80)
        if not AI_MANAGER.set_agent(self.agent_name):
            print(f" Failed to switch to agent {self.agent_name}")
            return
        if not self.agent_name.startswith("dqn"):
            print(f"  {self.agent_name} is not a trainable agent")
            return
        start_time = datetime.now()
        for episode in range(1, self.total_episodes + 1):
            episode_reward, episode_lines, episode_steps = self._run_episode(episode)
            self.training_rewards.append(episode_reward)
            self.training_lines.append(episode_lines)
            self.training_steps.append(episode_steps)
            if episode % 10 == 0:                self._print_progress(episode, start_time)
            if episode <= 20:
                agent_info = AI_MANAGER.get_agent_info()
                epsilon = agent_info.get('epsilon', 0.0)
                print(f"Episode {episode:3d}: Reward={episode_reward:6.1f}, Lines={episode_lines:2d}, Steps={episode_steps:3d}, Epsilon={epsilon:.3f}")
            if episode % self.eval_frequency == 0:
                eval_score = self._evaluate_agent(episode)
                self.eval_scores.append(eval_score)
                self.eval_episodes.append(episode)
                if eval_score > self.best_score:
                    self.best_score = eval_score
                    self.best_model_path = self._save_model(f"best_{self.agent_name}")
                    print(f" New best model! Score: {eval_score:.1f}")
            if episode % self.save_frequency == 0:
                self._save_model(f"{self.agent_name}_episode_{episode}")
                self._save_training_data()
                self._plot_training_progress()
        final_score = self._evaluate_agent(self.total_episodes, num_games=10)
        self._save_model(f"final_{self.agent_name}")
        self._save_training_data()
        self._plot_training_progress()
        print(f"\n Training completed!")
        print(f" Final evaluation score: {final_score:.1f}")
        print(f" Best score achieved: {self.best_score:.1f}")
        print(f"⏱  Total training time: {datetime.now() - start_time}")
        return {
            'final_score': final_score,
            'best_score': self.best_score,
            'training_rewards': self.training_rewards,
            'eval_scores': self.eval_scores
        }

    def _run_episode(self, episode):
        """Run a single training episode"""
        if episode <= 5:            print(f" Starting episode {episode}...")
        state = self.env.reset()
        total_reward = 0
        total_lines = 0
        steps = 0
        while not self.env.game_over and steps < self.max_steps_per_episode:
            action = AI_MANAGER.get_action(state, training=True)
            next_state, reward, done, info = self.env.step(action)
            AI_MANAGER.train_step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            total_lines += info.get('lines_cleared', 0)
            steps += 1
            if done:
                break
        if episode <= 5:            print(f" Episode {episode} completed: {steps} steps, reward: {total_reward:.1f}")
        return total_reward, total_lines, steps

    def _evaluate_agent(self, episode, num_games=5):
        """Evaluate agent performance without training"""
        print(f"\n Evaluating agent at episode {episode}...")
        eval_rewards = []
        eval_lines = []
        for game in range(num_games):
            state = self.env.reset()
            total_reward = 0
            total_lines = 0
            steps = 0
            while not self.env.game_over and steps < self.max_steps_per_episode:
                action = AI_MANAGER.get_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                total_reward += reward
                total_lines += info.get('lines_cleared', 0)
                steps += 1
                if done:
                    break
            eval_rewards.append(total_reward)
            eval_lines.append(total_lines)
        avg_reward = np.mean(eval_rewards)
        avg_lines = np.mean(eval_lines)
        print(f"   Avg Reward: {avg_reward:.1f} ± {np.std(eval_rewards):.1f}")
        print(f"   Avg Lines: {avg_lines:.1f} ± {np.std(eval_lines):.1f}")
        return avg_reward

    def _print_progress(self, episode, start_time):
        """Print training progress"""
        recent_rewards = self.training_rewards[-50:] if len(self.training_rewards) >= 50 else self.training_rewards
        recent_lines = self.training_lines[-50:] if len(self.training_lines) >= 50 else self.training_lines
        avg_reward = np.mean(recent_rewards)
        avg_lines = np.mean(recent_lines)
        agent_info = AI_MANAGER.get_agent_info()
        epsilon = agent_info.get('epsilon', 0.0)
        elapsed = datetime.now() - start_time
        episodes_per_hour = episode / (elapsed.total_seconds() / 3600)
        print(f"Episode {episode:4d} | "
              f"Avg Reward: {avg_reward:7.1f} | "
              f"Avg Lines: {avg_lines:5.1f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Speed: {episodes_per_hour:.1f} ep/h")

    def _save_model(self, model_name):
        """Save the current model"""
        filepath = os.path.join(config.MODEL_SAVE_PATH, f"{model_name}.pth")
        AI_MANAGER.save_current_model(filepath)
        return filepath

    def _save_training_data(self):
        """Save training metrics to file"""
        data = {
            'agent_name': self.agent_name,
            'total_episodes': len(self.training_rewards),
            'training_rewards': self.training_rewards,
            'training_lines': self.training_lines,
            'training_steps': self.training_steps,
            'eval_scores': self.eval_scores,
            'eval_episodes': self.eval_episodes,
            'best_score': self.best_score,
            'timestamp': datetime.now().isoformat()
        }
        filepath = os.path.join(config.LOG_PATH, f"training_data_{self.agent_name}.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f" Training data saved to {filepath}")

    def _plot_training_progress(self):
        """Plot and save training progress"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        def smooth_curve(data, window=50):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')

        episodes = range(1, len(self.training_rewards) + 1)
        ax1.plot(episodes, self.training_rewards, alpha=0.3, color='blue', label='Raw')
        if len(self.training_rewards) > 50:
            smooth_rewards = smooth_curve(self.training_rewards)
            smooth_episodes = range(25, 25 + len(smooth_rewards))
            ax1.plot(smooth_episodes, smooth_rewards, color='blue', linewidth=2, label='Smoothed')
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(episodes, self.training_lines, alpha=0.3, color='green', label='Raw')
        if len(self.training_lines) > 50:
            smooth_lines = smooth_curve(self.training_lines)
            smooth_episodes = range(25, 25 + len(smooth_lines))
            ax2.plot(smooth_episodes, smooth_lines, color='green', linewidth=2, label='Smoothed')
        ax2.set_title('Lines Cleared per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Lines Cleared')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        if self.eval_scores:
            ax3.plot(self.eval_episodes, self.eval_scores, 'ro-', linewidth=2, markersize=6)
            ax3.set_title('Evaluation Scores')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Average Evaluation Score')
            ax3.grid(True, alpha=0.3)
        ax4.plot(episodes, self.training_steps, alpha=0.3, color='orange', label='Raw')
        if len(self.training_steps) > 50:
            smooth_steps = smooth_curve(self.training_steps)
            smooth_episodes = range(25, 25 + len(smooth_steps))
            ax4.plot(smooth_episodes, smooth_steps, color='orange', linewidth=2, label='Smoothed')
        ax4.set_title('Episode Length (Steps)')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(config.LOG_PATH, f"training_progress_{self.agent_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Training plots saved to {plot_path}")


def train_multiple_agents():
    """Train multiple agents with different configurations"""
    agents_to_train = [
        ("dqn_basic", "Basic DQN with standard parameters"),
        ("dqn_advanced", "Advanced DQN with optimized parameters")
    ]
    results = {}
    for agent_name, description in agents_to_train:
        print(f"\n{'='*80}")
        print(f" Training {agent_name}: {description}")
        print(f"{'='*80}")
        trainer = TetrisTrainer(agent_name)
        results[agent_name] = trainer.train_agent()
        print(f"\n Completed training {agent_name}")
        print(f" Final Score: {results[agent_name]['final_score']:.1f}")
        print(f" Best Score: {results[agent_name]['best_score']:.1f}")
    print(f"\n{'='*80}")
    print(" TRAINING RESULTS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Agent':<15} {'Final Score':<12} {'Best Score':<12}")
    print("-" * 50)
    for agent_name, result in results.items():
        print(f"{agent_name:<15} {result['final_score']:<12.1f} {result['best_score']:<12.1f}")
    return results


def main():
    """Main training function"""
    print(" Tetris Deep Q-Network Training System")
    print("=" * 80)
    print(" Starting comprehensive training...")
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.LOG_PATH, exist_ok=True)
    results = train_multiple_agents()
    print(f"\n All training completed!")
    print(" You can now:")
    print("  - Run the game with trained AI agents")
    print("  - Use the API to test model performance")
    print("  - Run evaluation scripts to compare agents")
    print("  - View training plots in the logs directory")

if __name__ == "__main__":
    main()
