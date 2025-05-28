import sys
import argparse
from PyQt5.QtWidgets import QApplication
def main():
    parser = argparse.ArgumentParser(description='Tetris RL Project')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'play'], default='play',
                       help='Mode to run: train the AI, evaluate the AI, or play manually')
    parser.add_argument('--model', type=str, help='Path to model file for evaluation')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes for training')
    args = parser.parse_args()
    if args.mode == 'train':
        from scripts.train import train_agent
        train_agent(episodes=args.episodes)
    elif args.mode == 'evaluate':
        from scripts.evaluate import evaluate_agent
        if not args.model:
            print("Please provide a model path with --model for evaluation")
            return
        evaluate_agent(args.model)
    elif args.mode == 'play':
        from ui.game_UI import Tetris
        app = QApplication([])
        tetris = Tetris()
        sys.exit(app.exec_())
if __name__ == '__main__':
    main()
