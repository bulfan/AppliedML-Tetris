#Entry point: training loop, ε-greedy schedule, evaluation logs
#• As this grows, you may split into train.py, evaluate.py,
#  and a config.py (hyperparameters).

import sys
from PyQt5.QtWidgets import QApplication
from game_UI import Tetris

if __name__ == '__main__':
    app = QApplication([])
    tetris = Tetris()
    sys.exit(app.exec_())
