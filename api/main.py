"""
Tetris RL Model API

A RESTful API for serving the trained Tetris Deep Q-Network (DQN) agent.
This API allows users to get AI recommendations for Tetris moves based on the current game state.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np
import torch
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dqn_agent import DQNAgent
from utils.preprocessing import TetrisPreprocessor
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Tetris RL Model API",
    description="""
    A RESTful API for the Tetris Deep Q-Network (DQN) agent.
    
    This API provides intelligent move recommendations for Tetris gameplay based on 
    a trained reinforcement learning model. The model analyzes the current board state 
    and returns the best action to take along with confidence scores.
    
    ## Features
    - **Move Prediction**: Get AI recommendations for the next best move
    - **Board Analysis**: Analyze board state features and statistics
    - **Model Information**: Access model metadata and performance metrics
    - **Health Checks**: Monitor API status and model availability
    
    ## Actions
    - **0**: Move Left - Move the current piece one position to the left
    - **1**: Move Right - Move the current piece one position to the right  
    - **2**: Rotate - Rotate the current piece clockwise
    - **3**: Drop - Hard drop the current piece to the bottom
    """,
    version="1.0.0",
    contact={
        "name": "Tetris RL Team",
        "email": "tetris-rl@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model_agent: Optional[DQNAgent] = None
preprocessor: Optional[TetrisPreprocessor] = None
model_loaded = False
model_path = None

# Pydantic models for request/response validation
class BoardState(BaseModel):
    """
    Represents the current state of the Tetris board.
    
    The board is represented as a 2D array where:
    - 0 represents an empty cell
    - 1-7 represent different tetromino pieces
    """
    board: List[List[int]] = Field(
        ...,
        description="2D array representing the Tetris board state. Must be 22 rows x 10 columns.",
        example=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # ... (20 more rows)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            
            [1, 1, 0, 0, 0, 0, 0, 2, 2, 2]
        ]
    )
    
    @validator('board')
    def validate_board_dimensions(cls, v):
        if len(v) != config.BOARD_HEIGHT:
            raise ValueError(f'Board must have exactly {config.BOARD_HEIGHT} rows')
        
        for i, row in enumerate(v):
            if len(row) != config.BOARD_WIDTH:
                raise ValueError(f'Row {i} must have exactly {config.BOARD_WIDTH} columns')
            
            for j, cell in enumerate(row):
                if not isinstance(cell, int) or cell < 0 or cell > 7:
                    raise ValueError(f'Cell [{i}][{j}] must be an integer between 0 and 7')
        
        return v

class PredictionRequest(BaseModel):
    """Request model for move prediction."""
    board_state: BoardState = Field(..., description="Current state of the Tetris board")
    include_analysis: bool = Field(
        default=False, 
        description="Whether to include detailed board analysis in the response"
    )

class ActionPrediction(BaseModel):
    """Model for action prediction with confidence."""
    action: int = Field(..., description="Predicted action ID (0=left, 1=right, 2=rotate, 3=drop)")
    action_name: str = Field(..., description="Human-readable action name")
    confidence: float = Field(..., description="Confidence score for this action (0-1)")
    q_value: float = Field(..., description="Q-value for this action")

class BoardAnalysis(BaseModel):
    """Detailed analysis of the board state."""
    column_heights: List[int] = Field(..., description="Height of each column")
    total_holes: int = Field(..., description="Number of holes in the board")
    bumpiness: float = Field(..., description="Measure of height variation between columns")
    complete_lines: int = Field(..., description="Number of complete lines ready to clear")
    max_height: int = Field(..., description="Maximum column height")
    avg_height: float = Field(..., description="Average column height")
    wells: int = Field(..., description="Number of wells (deep single-column gaps)")

class PredictionResponse(BaseModel):
    """Response model for move prediction."""
    model_config = {"protected_namespaces": ()}
    
    success: bool = Field(..., description="Whether the prediction was successful")
    recommended_action: ActionPrediction = Field(..., description="The recommended action to take")
    all_actions: List[ActionPrediction] = Field(..., description="Q-values for all possible actions")
    board_analysis: Optional[BoardAnalysis] = Field(None, description="Detailed board analysis (if requested)")
    timestamp: datetime = Field(..., description="Timestamp of the prediction")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")

class ModelInfo(BaseModel):
    """Information about the loaded model."""
    model_config = {"protected_namespaces": ()}
    
    model_loaded: bool = Field(..., description="Whether a model is currently loaded")
    model_path: Optional[str] = Field(None, description="Path to the loaded model file")
    model_architecture: Dict[str, Any] = Field(..., description="Model architecture details")
    feature_size: int = Field(..., description="Size of the input feature vector")
    action_space_size: int = Field(..., description="Number of possible actions")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")

# Action mapping
ACTION_NAMES = {
    0: "move_left",
    1: "move_right", 
    2: "rotate",
    3: "drop"
}

def load_model(model_file_path: str = None):
    """Load the trained model and preprocessor."""
    global model_agent, preprocessor, model_loaded, model_path
    
    try:
        logger.info("Initializing preprocessor...")
        preprocessor = TetrisPreprocessor()
        
        # Initialize with dummy data to fit the preprocessor
        dummy_board = np.zeros((config.BOARD_HEIGHT, config.BOARD_WIDTH))
        dummy_features = [preprocessor.extract_features(dummy_board) for _ in range(10)]
        preprocessor.fit_transform(dummy_features)
        
        # Initialize agent
        state_size = preprocessor.get_feature_size()
        action_size = 4
        logger.info(f"Initializing DQN agent with state_size={state_size}, action_size={action_size}")
        model_agent = DQNAgent(state_size, action_size)
        
        # Try to load a trained model if available
        if model_file_path and os.path.exists(model_file_path):
            logger.info(f"Loading model from {model_file_path}")
            model_agent.load(model_file_path)
            model_path = model_file_path
            logger.info(f"Model loaded successfully from {model_file_path}")
        else:
            # Look for any available trained models
            model_dir = config.MODEL_SAVE_PATH
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                if model_files:
                    # Load the first available model
                    model_file = os.path.join(model_dir, model_files[0])
                    logger.info(f"Loading available model: {model_file}")
                    model_agent.load(model_file)
                    model_path = model_file
                    logger.info(f"Model loaded from {model_file}")
                else:
                    logger.warning("No trained model found. Using untrained model.")
                    model_path = None
            else:
                logger.warning("No models directory found. Using untrained model.")
                model_path = None
        
        model_loaded = True
        logger.info("Model and preprocessor initialized successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loaded = False
        # Don't raise the exception - let the API start with an untrained model
        logger.warning("API will start with untrained model")

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Tetris RL Model API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API and model status.
    
    Returns the current status of the API and whether the model is loaded and ready to serve predictions.
    """
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the currently loaded model.
    
    Returns details about the model architecture, feature size, and loading status.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check the model file path and try again."
        )
    
    return ModelInfo(
        model_loaded=model_loaded,
        model_path=model_path,
        model_architecture={
            "type": "Deep Q-Network (DQN)",
            "hidden_layers": config.HIDDEN_LAYERS,
            "dropout_rate": config.DROPOUT_RATE,
            "learning_rate": config.LEARNING_RATE
        },
        feature_size=preprocessor.get_feature_size(),
        action_space_size=4
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_move(request: PredictionRequest):
    """
    Predict the best move for the given Tetris board state.
    
    This endpoint analyzes the current board state and returns the AI's recommended action
    along with confidence scores for all possible actions.
    
    **Parameters:**
    - **board_state**: 2D array (22x10) representing the current Tetris board
    - **include_analysis**: Optional flag to include detailed board analysis
    
    **Returns:**
    - **recommended_action**: The best action to take with confidence score
    - **all_actions**: Q-values and confidence for all possible actions
    - **board_analysis**: Detailed board statistics (if requested)
    - **model_info**: Information about the model used for prediction
    
    **Example board state:**
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
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure the model is properly initialized."
        )
    
    try:
        # Convert board to numpy array
        board_array = np.array(request.board_state.board)
        
        # Extract features
        features = preprocessor.extract_features(board_array)
        state_features = preprocessor.transform(features)
        
        # Get Q-values for all actions
        with torch.no_grad():
            q_values = model_agent.q_network(torch.FloatTensor(state_features).unsqueeze(0))
            q_values = q_values.cpu().numpy().flatten()
        
        # Get recommended action
        recommended_action_id = np.argmax(q_values)
        
        # Convert Q-values to confidence scores (softmax)
        exp_q = np.exp(q_values - np.max(q_values))  # Numerical stability
        confidences = exp_q / np.sum(exp_q)
        
        # Create action predictions
        all_actions = []
        for i, (q_val, conf) in enumerate(zip(q_values, confidences)):
            all_actions.append(ActionPrediction(
                action=i,
                action_name=ACTION_NAMES[i],
                confidence=float(conf),
                q_value=float(q_val)
            ))
        
        recommended_action = all_actions[recommended_action_id]
        
        # Board analysis (if requested)
        board_analysis = None
        if request.include_analysis:
            heights = preprocessor._get_column_heights(board_array)
            holes = preprocessor._count_holes(board_array)
            complete_lines = preprocessor._count_complete_lines(board_array)
            wells = preprocessor._count_wells(board_array)
            
            height_diffs = [heights[i+1] - heights[i] for i in range(len(heights)-1)]
            bumpiness = sum(abs(diff) for diff in height_diffs)
            
            board_analysis = BoardAnalysis(
                column_heights=heights,
                total_holes=holes,
                bumpiness=float(bumpiness),
                complete_lines=complete_lines,
                max_height=max(heights),
                avg_height=float(np.mean(heights)),
                wells=wells
            )
        
        return PredictionResponse(
            success=True,
            recommended_action=recommended_action,
            all_actions=all_actions,
            board_analysis=board_analysis,
            timestamp=datetime.now(),
            model_info={
                "model_type": "DQN",
                "feature_size": preprocessor.get_feature_size(),
                "model_path": model_path or "untrained"
            }
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing prediction: {str(e)}"
        )

@app.post("/model/load")
async def load_model_endpoint(model_path: str):
    """
    Load a specific model file.
    
    **Parameters:**
    - **model_path**: Path to the model file to load
    
    **Returns:**
    - Success message and model information
    """
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file not found: {model_path}"
        )
    
    try:
        load_model(model_path)
        return {
            "success": True,
            "message": f"Model loaded successfully from {model_path}",
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 