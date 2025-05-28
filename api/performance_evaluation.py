"""
Performance Evaluation Script for Tetris RL Model
This script evaluates the model performance against random baseline
to demonstrate that the model performs above random guessing.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from typing import List, Dict, Tuple
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent
from utils.preprocessing import TetrisPreprocessor
import config
class PerformanceEvaluator:
    """Evaluates model performance against baselines."""
    def __init__(self):
        self.env = TetrisEnv()
        self.preprocessor = TetrisPreprocessor()
        self.results = {}
    def evaluate_random_baseline(self, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate random action baseline."""
        print(f" Evaluating random baseline ({num_episodes} episodes)...")
        scores = []
        lines_cleared = []
        game_lengths = []
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            total_lines = 0
            steps = 0
            while not self.env.game_over and steps < config.MAX_STEPS_PER_EPISODE:
                action = np.random.randint(0, 4)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                total_lines += info.get('lines_cleared', 0)
                steps += 1
            scores.append(total_reward)
            lines_cleared.append(total_lines)
            game_lengths.append(steps)
            if episode % 20 == 0:
                print(f"   Episode {episode}: Score {total_reward:.1f}, Lines {total_lines}")
        results = {
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'avg_lines': np.mean(lines_cleared),
            'std_lines': np.std(lines_cleared),
            'avg_length': np.mean(game_lengths),
            'scores': scores,
            'lines_cleared': lines_cleared,
            'game_lengths': game_lengths
        }
        print(f"    Random Baseline Results:")
        print(f"      Average Score: {results['avg_score']:.2f} ± {results['std_score']:.2f}")
        print(f"      Average Lines: {results['avg_lines']:.2f} ± {results['std_lines']:.2f}")
        print(f"      Average Length: {results['avg_length']:.2f} steps")
        return results
    def evaluate_model(self, model_path: str = None, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate the trained model."""
        print(f" Evaluating trained model ({num_episodes} episodes)...")
        state_size = self.preprocessor.get_feature_size()
        action_size = 4
        agent = DQNAgent(state_size, action_size)
        if model_path and os.path.exists(model_path):
            agent.load(model_path)
            print(f"   Loaded model from {model_path}")
        else:
            default_path = os.path.join(config.MODEL_SAVE_PATH, "tetris_dqn_final.pth")
            if os.path.exists(default_path):
                agent.load(default_path)
                print(f"   Loaded default model from {default_path}")
            else:
                print("     No trained model found, using untrained model")
        agent.epsilon = 0.0
        dummy_board = np.zeros((config.BOARD_HEIGHT, config.BOARD_WIDTH))
        dummy_features = [self.preprocessor.extract_features(dummy_board) for _ in range(10)]
        self.preprocessor.fit_transform(dummy_features)
        scores = []
        lines_cleared = []
        game_lengths = []
        for episode in range(num_episodes):
            state = self.env.reset()
            features = self.preprocessor.extract_features(state)
            state_features = self.preprocessor.transform(features)
            total_reward = 0
            total_lines = 0
            steps = 0
            while not self.env.game_over and steps < config.MAX_STEPS_PER_EPISODE:
                action = agent.act(state_features)
                next_state, reward, done, info = self.env.step(action)
                next_features = self.preprocessor.extract_features(next_state)
                state_features = self.preprocessor.transform(next_features)
                total_reward += reward
                total_lines += info.get('lines_cleared', 0)
                steps += 1
            scores.append(total_reward)
            lines_cleared.append(total_lines)
            game_lengths.append(steps)
            if episode % 20 == 0:
                print(f"   Episode {episode}: Score {total_reward:.1f}, Lines {total_lines}")
        results = {
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'avg_lines': np.mean(lines_cleared),
            'std_lines': np.std(lines_cleared),
            'avg_length': np.mean(game_lengths),
            'scores': scores,
            'lines_cleared': lines_cleared,
            'game_lengths': game_lengths
        }
        print(f"    Model Results:")
        print(f"      Average Score: {results['avg_score']:.2f} ± {results['std_score']:.2f}")
        print(f"      Average Lines: {results['avg_lines']:.2f} ± {results['std_lines']:.2f}")
        print(f"      Average Length: {results['avg_length']:.2f} steps")
        return results
    def evaluate_heuristic_baseline(self, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate a simple heuristic baseline (always drop)."""
        print(f" Evaluating heuristic baseline - always drop ({num_episodes} episodes)...")
        scores = []
        lines_cleared = []
        game_lengths = []
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            total_lines = 0
            steps = 0
            while not self.env.game_over and steps < config.MAX_STEPS_PER_EPISODE:
                action = 3
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                total_lines += info.get('lines_cleared', 0)
                steps += 1
            scores.append(total_reward)
            lines_cleared.append(total_lines)
            game_lengths.append(steps)
            if episode % 20 == 0:
                print(f"   Episode {episode}: Score {total_reward:.1f}, Lines {total_lines}")
        results = {
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'avg_lines': np.mean(lines_cleared),
            'std_lines': np.std(lines_cleared),
            'avg_length': np.mean(game_lengths),
            'scores': scores,
            'lines_cleared': lines_cleared,
            'game_lengths': game_lengths
        }
        print(f"    Heuristic Baseline Results:")
        print(f"      Average Score: {results['avg_score']:.2f} ± {results['std_score']:.2f}")
        print(f"      Average Lines: {results['avg_lines']:.2f} ± {results['std_lines']:.2f}")
        print(f"      Average Length: {results['avg_length']:.2f} steps")
        return results
    def compare_performance(self, model_path: str = None, num_episodes: int = 100) -> Dict[str, Dict]:
        """Compare model performance against baselines."""
        print(" Performance Comparison")
        print("=" * 60)
        random_results = self.evaluate_random_baseline(num_episodes)
        heuristic_results = self.evaluate_heuristic_baseline(num_episodes)
        model_results = self.evaluate_model(model_path, num_episodes)
        self.results = {
            'random': random_results,
            'heuristic': heuristic_results,
            'model': model_results,
            'evaluation_date': datetime.now().isoformat(),
            'num_episodes': num_episodes
        }
        score_improvement_vs_random = (
            (model_results['avg_score'] - random_results['avg_score']) / 
            abs(random_results['avg_score']) * 100
        )
        lines_improvement_vs_random = (
            (model_results['avg_lines'] - random_results['avg_lines']) / 
            max(random_results['avg_lines'], 0.1) * 100
        )
        score_improvement_vs_heuristic = (
            (model_results['avg_score'] - heuristic_results['avg_score']) / 
            abs(heuristic_results['avg_score']) * 100
        )
        print("\n PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"{'Method':<15} {'Avg Score':<12} {'Avg Lines':<12} {'Avg Length':<12}")
        print("-" * 60)
        print(f"{'Random':<15} {random_results['avg_score']:<12.2f} {random_results['avg_lines']:<12.2f} {random_results['avg_length']:<12.2f}")
        print(f"{'Heuristic':<15} {heuristic_results['avg_score']:<12.2f} {heuristic_results['avg_lines']:<12.2f} {heuristic_results['avg_length']:<12.2f}")
        print(f"{'Model':<15} {model_results['avg_score']:<12.2f} {model_results['avg_lines']:<12.2f} {model_results['avg_length']:<12.2f}")
        print("\n IMPROVEMENTS")
        print("=" * 60)
        print(f"Model vs Random:")
        print(f"  Score: {score_improvement_vs_random:+.1f}%")
        print(f"  Lines: {lines_improvement_vs_random:+.1f}%")
        print(f"Model vs Heuristic:")
        print(f"  Score: {score_improvement_vs_heuristic:+.1f}%")
        from scipy import stats
        try:
            t_stat_score, p_val_score = stats.ttest_ind(
                model_results['scores'], 
                random_results['scores']
            )
            t_stat_lines, p_val_lines = stats.ttest_ind(
                model_results['lines_cleared'], 
                random_results['lines_cleared']
            )
            print(f"\n STATISTICAL SIGNIFICANCE")
            print("=" * 60)
            print(f"Score difference p-value: {p_val_score:.6f}")
            print(f"Lines difference p-value: {p_val_lines:.6f}")
            if p_val_score < 0.05:
                print(" Score improvement is statistically significant (p < 0.05)")
            else:
                print("  Score improvement is not statistically significant")
            if p_val_lines < 0.05:
                print(" Lines improvement is statistically significant (p < 0.05)")
            else:
                print("  Lines improvement is not statistically significant")
        except ImportError:
            print("\n  scipy not available for statistical tests")
        return self.results
    def plot_results(self, save_path: str = None):
        """Plot comparison results."""
        if not self.results:
            print("No results to plot. Run comparison first.")
            return
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        methods = ['Random', 'Heuristic', 'Model']
        scores = [self.results['random']['avg_score'], 
                 self.results['heuristic']['avg_score'], 
                 self.results['model']['avg_score']]
        lines = [self.results['random']['avg_lines'], 
                self.results['heuristic']['avg_lines'], 
                self.results['model']['avg_lines']]
        ax1.bar(methods, scores, color=['red', 'orange', 'green'], alpha=0.7)
        ax1.set_title('Average Score Comparison')
        ax1.set_ylabel('Average Score')
        ax1.grid(True, alpha=0.3)
        ax2.bar(methods, lines, color=['red', 'orange', 'green'], alpha=0.7)
        ax2.set_title('Average Lines Cleared Comparison')
        ax2.set_ylabel('Average Lines Cleared')
        ax2.grid(True, alpha=0.3)
        ax3.hist(self.results['random']['scores'], alpha=0.5, label='Random', bins=20, color='red')
        ax3.hist(self.results['model']['scores'], alpha=0.5, label='Model', bins=20, color='green')
        ax3.set_title('Score Distribution')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax4.hist(self.results['random']['lines_cleared'], alpha=0.5, label='Random', bins=20, color='red')
        ax4.hist(self.results['model']['lines_cleared'], alpha=0.5, label='Model', bins=20, color='green')
        ax4.set_title('Lines Cleared Distribution')
        ax4.set_xlabel('Lines Cleared')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Plot saved to {save_path}")
        plt.show()
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        if not self.results:
            print("No results to save. Run comparison first.")
            return
        results_copy = {}
        for method, data in self.results.items():
            if isinstance(data, dict):
                results_copy[method] = {}
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        results_copy[method][key] = list(value)
                    else:
                        results_copy[method][key] = value
            else:
                results_copy[method] = data
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2)
        print(f" Results saved to {filepath}")
def main():
    """Main evaluation function."""
    print(" Starting Tetris RL Performance Evaluation")
    print("=" * 60)
    evaluator = PerformanceEvaluator()
    results = evaluator.compare_performance(num_episodes=50)    
    plot_path = os.path.join(config.LOG_PATH, 'performance_comparison.png')
    evaluator.plot_results(plot_path)
    results_path = os.path.join(config.LOG_PATH, 'performance_results.json')
    evaluator.save_results(results_path)
    print("\n Evaluation completed!")
    print(f" Plots saved to: {plot_path}")
    print(f" Results saved to: {results_path}")
if __name__ == "__main__":
    main() 