# Tetris RL - Deep Q-Network Agent

A reinforcement learning project that trains a Deep Q-Network (DQN) agent to play Tetris using PyTorch and PyQt5.

## Project Structure

```
tetris_rl/
├── env/
│   ├── game_data.py          # Core game logic and data structures
│   └── tetris_env.py         # RL environment wrapper
├── agents/
│   ├── model.py              # Neural network architecture
│   ├── replay_buffer.py      # Experience replay buffer
│   └── dqn_agent.py          # DQN agent implementation
├── scripts/
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation script
├── ui/
│   └── game_UI.py            # PyQt5 game interface
├── utils/
│   └── preprocessing.py      # Feature extraction and preprocessing
├── api/
│   ├── main.py               # FastAPI application
│   ├── test_client.py        # API testing client
│   └── performance_evaluation.py  # Model performance evaluation
├── config.py                 # Configuration and hyperparameters
├── run.py                    # Main entry point
├── start_api.py              # API server startup script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Features

- **Deep Q-Network (DQN)** with experience replay and target networks
- **Feature Engineering**: Extracts meaningful features from game state (heights, holes, bumpiness, etc.)
- **Preprocessing**: Optional PCA and normalization for feature scaling
- **GUI Interface**: Play manually or watch the AI play
- **Training Visualization**: Real-time training metrics and plots
- **Model Evaluation**: Comprehensive evaluation with statistics and visualizations
- **🆕 REST API**: Deploy model as a web service with comprehensive documentation

## Setup

### 1. Create Virtual Environment

```bash
python -m venv tetris_env
source tetris_env/bin/activate  # On Windows: tetris_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Game

```bash
# Play manually (default)
python run.py

# Train the AI
python run.py --mode train --episodes 1000

# Evaluate a trained model
python run.py --mode evaluate --model models/tetris_dqn_final.pth
```

## API Deployment

### Starting the API Server

```bash
# Method 1: Using the startup script (recommended)
python start_api.py

# Method 2: Using uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Local**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check
```bash
GET /health
```
Check if the API and model are ready to serve requests.

#### 2. Model Information
```bash
GET /model/info
```
Get details about the loaded model architecture and configuration.

#### 3. Move Prediction (Main Endpoint)
```bash
POST /predict
```
Predict the best move for a given Tetris board state.

**Request Format:**
```json
{
  "board_state": {
    "board": [
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      ...
      [1, 1, 0, 0, 0, 0, 0, 2, 2, 2]
    ]
  },
  "include_analysis": true
}
```

**Response Format:**
```json
{
  "success": true,
  "recommended_action": {
    "action": 2,
    "action_name": "rotate",
    "confidence": 0.85,
    "q_value": 1.23
  },
  "all_actions": [...],
  "board_analysis": {
    "column_heights": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    "total_holes": 0,
    "bumpiness": 2.0,
    "complete_lines": 0,
    "max_height": 1,
    "avg_height": 0.3,
    "wells": 0
  },
  "timestamp": "2024-01-15T10:30:00",
  "model_info": {...}
}
```

### Testing the API

#### Using curl
```bash
# Health check
curl -X GET http://localhost:8000/health

# Model information
curl -X GET http://localhost:8000/model/info

# Move prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "board_state": {
      "board": [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
    },
    "include_analysis": false
  }'
```

#### Using Python Test Client
```bash
python api/test_client.py
```

### API Features

#### ✅ RESTful Design
- **GET** endpoints for retrieving information (health, model info)
- **POST** endpoints for predictions and actions
- Proper HTTP status codes and error handling
- JSON request/response format
- Stateless operations

#### ✅ Comprehensive Documentation
- **OpenAPI/Swagger** documentation at `/docs`
- **ReDoc** alternative documentation at `/redoc`
- Detailed endpoint descriptions with examples
- Request/response schema validation with Pydantic
- Natural language explanations for all endpoints

#### ✅ Error Handling
- **422 Validation Error**: Invalid board dimensions or cell values
- **503 Service Unavailable**: Model not loaded
- **500 Internal Server Error**: Processing errors
- **404 Not Found**: Model file not found
- Detailed error messages with specific problem descriptions

#### ✅ Input Preprocessing
- Automatic board validation (22x10 dimensions, valid cell values)
- Feature extraction and normalization handled internally
- No manual preprocessing required from users
- Converts raw board state to model-ready features

#### ✅ Output Postprocessing
- Returns human-readable action names instead of just IDs
- Confidence scores calculated from Q-values
- Optional detailed board analysis
- Structured response with metadata

## Model Performance

### Performance Evaluation

Run the performance evaluation to verify the model performs above random baseline:

```bash
python api/performance_evaluation.py
```

This will:
- Evaluate random baseline performance
- Evaluate heuristic baseline (always drop)
- Evaluate trained model performance
- Generate statistical comparison
- Create visualization plots
- Save results to JSON

### Expected Results

The trained model should demonstrate:
- **Higher average scores** than random actions
- **More lines cleared** than random actions
- **Longer game duration** indicating better survival
- **Statistical significance** (p < 0.05) in performance improvements

## Usage

### Manual Play

```bash
python run.py --mode play
```

**Controls:**
- ← → : Move left/right
- ↑ : Rotate piece
- Space : Drop piece
- P : Pause game
- R : Restart game

### Training the AI

```bash
python run.py --mode train --episodes 1000
```

This will:
- Train a DQN agent for the specified number of episodes
- Save models periodically to `models/` directory
- Generate training plots in `logs/` directory
- Print training progress and metrics

### Evaluating the AI

```bash
python run.py --mode evaluate --model models/tetris_dqn_final.pth --episodes 100
```

This will:
- Load the specified model
- Run evaluation episodes without exploration
- Generate evaluation statistics and plots
- Show performance metrics

## Configuration

Edit `config.py` to modify:

### Training Parameters
- `MAX_EPISODES`: Number of training episodes
- `LEARNING_RATE`: Neural network learning rate
- `EPSILON_START/END/DECAY`: Exploration parameters
- `BATCH_SIZE`: Training batch size
- `MEMORY_SIZE`: Replay buffer size

### Network Architecture
- `HIDDEN_LAYERS`: List of hidden layer sizes
- `DROPOUT_RATE`: Dropout probability

### Rewards
- `REWARD_LINE_CLEAR`: Points for clearing lines
- `REWARD_GAME_OVER`: Penalty for game over
- `REWARD_STEP`: Small penalty per step

## Model Architecture

The DQN uses a fully connected neural network:
- Input: Feature vector (board state features)
- Hidden layers: Configurable (default: [512, 256, 128])
- Output: Q-values for 4 actions (left, right, rotate, drop)
- Activation: ReLU
- Dropout: Configurable rate

## Feature Engineering

The state representation includes:
- **Column heights** (10 features)
- **Height differences** between adjacent columns (9 features)
- **Aggregate statistics**: max height, mean height, height variance
- **Holes**: Empty cells with filled cells above
- **Bumpiness**: Sum of absolute height differences
- **Complete lines**: Lines ready to be cleared
- **Wells**: Deep single-column gaps
- **Blocked cells**: Cells with empty cells above

## Training Process

1. **Experience Collection**: Agent interacts with environment using ε-greedy policy
2. **Experience Replay**: Random sampling from replay buffer for training
3. **Target Network**: Separate target network updated periodically
4. **Feature Preprocessing**: Optional PCA and normalization
5. **Progress Tracking**: Metrics logged and visualized

## Results and Visualization

Training and evaluation generate several plots:
- Training rewards over episodes
- Lines cleared over episodes
- Score distributions
- Game length distributions
- Model comparison charts

## Advanced Usage

### Custom Training Script

```python
from scripts.train import train_agent
from config import MAX_EPISODES

# Train with custom parameters
agent = train_agent(episodes=MAX_EPISODES)
```

### Model Comparison

```python
from scripts.evaluate import compare_models

models = {
    'Model_1000': 'models/tetris_dqn_episode_1000.pth',
    'Model_2000': 'models/tetris_dqn_episode_2000.pth',
    'Final': 'models/tetris_dqn_final.pth'
}

results = compare_models(models, num_episodes=50)
```

### Custom Feature Engineering

```python
from utils.preprocessing import TetrisPreprocessor

preprocessor = TetrisPreprocessor(use_pca=True, n_components=30)
features = preprocessor.extract_features(board_state)
```

### API Integration

```python
from api.test_client import TetrisAPIClient

client = TetrisAPIClient("http://localhost:8000")
result = client.predict_move(board_state, include_analysis=True)
print(f"Recommended action: {result['recommended_action']['action_name']}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the project root directory
2. **PyQt5 Issues**: Install PyQt5 separately if needed: `pip install PyQt5`
3. **CUDA Issues**: Set `device = 'cpu'` in config.py if GPU issues occur
4. **Memory Issues**: Reduce `MEMORY_SIZE` or `BATCH_SIZE` in config.py
5. **API Port Issues**: Change port in `start_api.py` if 8000 is occupied

### Performance Tips

- Use GPU if available (set `device = 'cuda'` in config.py)
- Adjust `TARGET_UPDATE_FREQUENCY` for stability
- Tune `EPSILON_DECAY` for exploration/exploitation balance
- Experiment with different network architectures

### API Troubleshooting

- **503 Service Unavailable**: Ensure model is trained and available
- **422 Validation Error**: Check board dimensions (22x10) and cell values (0-7)
- **Connection Refused**: Verify API server is running on correct port

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Based on the classic Tetris game
- Inspired by DeepMind's DQN paper
- Uses PyQt5 for the graphical interface
- FastAPI for REST API deployment
