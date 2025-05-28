import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from env.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent
from utils.preprocessing import TetrisPreprocessor
def evaluate_agent(model_path, num_episodes=100, render=False):
    """Evaluate a trained DQN agent"""
    env = TetrisEnv()
    preprocessor = TetrisPreprocessor()
    state_size = preprocessor.get_feature_size()
    action_size = 4    
    agent = DQNAgent(state_size, action_size)
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found!")
        return
    agent.epsilon = 0.0
    scores = []
    lines_cleared_list = []
    game_lengths = []
    print(f"Evaluating agent for {num_episodes} episodes...")
    for episode in range(num_episodes):
        state = env.reset()
        features = preprocessor.extract_features(state)
        if not preprocessor.fitted:
            initial_features = [features]
            for _ in range(50):
                temp_state = env.reset()
                temp_features = preprocessor.extract_features(temp_state)
                initial_features.append(temp_features)
            preprocessor.fit_transform(initial_features)
        state_features = preprocessor.transform(features)
        total_reward = 0
        total_lines = 0
        steps = 0
        while not env.game_over and steps < config.MAX_STEPS_PER_EPISODE:
            action = agent.act(state_features)
            next_state, reward, done, info = env.step(action)
            next_features = preprocessor.extract_features(next_state)
            next_state_features = preprocessor.transform(next_features)
            state_features = next_state_features
            total_reward += reward
            total_lines += info.get('lines_cleared', 0)
            steps += 1
            if render and episode < 5:                env.render()
        scores.append(total_reward)
        lines_cleared_list.append(total_lines)
        game_lengths.append(steps)
        if episode % 10 == 0:
            print(f"Episode {episode}: Score {total_reward:.2f}, Lines {total_lines}, Steps {steps}")
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    avg_lines = np.mean(lines_cleared_list)
    std_lines = np.std(lines_cleared_list)
    avg_length = np.mean(game_lengths)
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {avg_score:.2f} ± {std_score:.2f}")
    print(f"Average Lines Cleared: {avg_lines:.2f} ± {std_lines:.2f}")
    print(f"Average Game Length: {avg_length:.2f} steps")
    print(f"Best Score: {max(scores):.2f}")
    print(f"Best Lines Cleared: {max(lines_cleared_list)}")
    print("="*50)
    plot_evaluation_results(scores, lines_cleared_list, game_lengths)
    return {
        'scores': scores,
        'lines_cleared': lines_cleared_list,
        'game_lengths': game_lengths,
        'avg_score': avg_score,
        'avg_lines': avg_lines
    }
def plot_evaluation_results(scores, lines_cleared, game_lengths):
    """Plot evaluation metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    ax1.plot(scores)
    ax1.set_title('Episode Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax2.plot(lines_cleared)
    ax2.set_title('Lines Cleared per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Lines Cleared')
    ax2.grid(True)
    ax3.hist(scores, bins=20, alpha=0.7)
    ax3.set_title('Score Distribution')
    ax3.set_xlabel('Score')
    ax3.set_ylabel('Frequency')
    ax3.grid(True)
    ax4.hist(game_lengths, bins=20, alpha=0.7)
    ax4.set_title('Game Length Distribution')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Frequency')
    ax4.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOG_PATH, 'evaluation_results.png'))
    plt.show()
def compare_models(model_paths, num_episodes=50):
    """Compare multiple trained models"""
    results = {}
    for model_name, model_path in model_paths.items():
        print(f"\nEvaluating {model_name}...")
        result = evaluate_agent(model_path, num_episodes)
        results[model_name] = result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    model_names = list(results.keys())
    avg_scores = [results[name]['avg_score'] for name in model_names]
    avg_lines = [results[name]['avg_lines'] for name in model_names]
    ax1.bar(model_names, avg_scores)
    ax1.set_title('Average Scores Comparison')
    ax1.set_ylabel('Average Score')
    ax1.tick_params(axis='x', rotation=45)
    ax2.bar(model_names, avg_lines)
    ax2.set_title('Average Lines Cleared Comparison')
    ax2.set_ylabel('Average Lines Cleared')
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOG_PATH, 'model_comparison.png'))
    plt.show()
    return results
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Tetris DQN Agent')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render some episodes')
    args = parser.parse_args()
    evaluate_agent(args.model, args.episodes, args.render)
