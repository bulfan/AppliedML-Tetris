# AppliedML-Tetris
We will use the environment from this git:
https://github.com/LoveDaisy/tetris_game

FILES:
1. `game_data.py` the data used by the game and game UI
2. `game_UI.py` the game UI. Bassically everything you see or interact with in the game
3. `model.py` Where we will build the AI model that plays the game
4. `run.py` ties everything together. This is where you run the program from


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
│   └── game_data.py
├── agents/  
│   ├── model.py            # neural network  
│   ├── replay_buffer.py    # experience replay  
│   └── dqn_agent.py        # DQN logic (ε-greedy, target updates)  
├── scripts/  
│   ├── train.py  
│   └── evaluate.py  
├── ui/  
│   └── game_UI.py  
├── utils/  
│   └── preprocessing.py    # feature scaling, PCA  
├── README.md  
└── config.py               # all hyperparameters & paths  
