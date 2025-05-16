# AppliedML-Tetris
We will use the environment from this git:
https://github.com/LoveDaisy/tetris_game

HOW TO RUN:
1. Create a venv and install requirements
2. Run run.py
3. Play!

HOW TO PLAY:
left right arrow keys to move the tetromino
up arrow key to rotate
space to slam it down
p to pause game

TO DO:
1. Create model
2. Maybe tweak UI 
3. Maybe add more functionality to UI



Proposed final structure?
tetris_rl/
├── env/  
│   └── tetris_env.py       # wraps Gym or your own logic  
├── agents/  
│   ├── model.py            # neural network  
│   ├── replay_buffer.py    # experience replay  
│   └── dqn_agent.py        # DQN logic (ε-greedy, target updates)  
├── scripts/  
│   ├── train.py  
│   └── evaluate.py  
├── ui/  
│   └── game_ui.py  
├── utils/  
│   └── preprocessing.py    # feature scaling, PCA  
├── README.md  
└── config.py               # all hyperparameters & paths  
