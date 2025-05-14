# AppliedML-Tetris

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
