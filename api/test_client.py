"""
Test client for the Tetris RL API

This script demonstrates how to interact with the API and provides examples
for testing the endpoints.
"""

import requests
import json
import numpy as np
from typing import Dict, Any

class TetrisAPIClient:
    """Client for interacting with the Tetris RL API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        response = self.session.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()
    
    def predict_move(self, board_state: list, include_analysis: bool = False) -> Dict[str, Any]:
        """Predict the best move for a given board state."""
        payload = {
            "board_state": {
                "board": board_state
            },
            "include_analysis": include_analysis
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Load a specific model file."""
        response = self.session.post(
            f"{self.base_url}/model/load",
            params={"model_path": model_path}
        )
        response.raise_for_status()
        return response.json()

def create_sample_board_states():
    """Create sample board states for testing."""
    
    # Empty board
    empty_board = [[0 for _ in range(10)] for _ in range(22)]
    
    # Board with some pieces at the bottom
    bottom_pieces_board = [[0 for _ in range(10)] for _ in range(22)]
    # Add some pieces at the bottom
    for row in range(19, 22):
        for col in range(0, 7):
            bottom_pieces_board[row][col] = np.random.randint(1, 8)
    
    # Board with holes
    holes_board = [[0 for _ in range(10)] for _ in range(22)]
    # Create a pattern with holes
    holes_board[21] = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]  # Bottom row with holes
    holes_board[20] = [1, 0, 1, 1, 1, 0, 1, 1, 0, 1]  # Second row with holes
    holes_board[19] = [0, 1, 1, 1, 0, 1, 1, 0, 1, 1]  # Third row with holes
    
    # Nearly full board (challenging scenario)
    full_board = [[0 for _ in range(10)] for _ in range(22)]
    for row in range(10, 22):
        for col in range(10):
            if np.random.random() > 0.3:  # 70% chance of having a piece
                full_board[row][col] = np.random.randint(1, 8)
    
    return {
        "empty": empty_board,
        "bottom_pieces": bottom_pieces_board,
        "with_holes": holes_board,
        "nearly_full": full_board
    }

def test_api_endpoints():
    """Test all API endpoints with sample data."""
    
    client = TetrisAPIClient()
    
    print("🎮 Testing Tetris RL API")
    print("=" * 50)
    
    # Test health check
    print("\n1. Health Check:")
    try:
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Model Loaded: {health['model_loaded']}")
        print(f"   Timestamp: {health['timestamp']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test model info
    print("\n2. Model Information:")
    try:
        model_info = client.get_model_info()
        print(f"   Model Type: {model_info['model_architecture']['type']}")
        print(f"   Feature Size: {model_info['feature_size']}")
        print(f"   Action Space: {model_info['action_space_size']}")
        print(f"   Model Path: {model_info['model_path']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test predictions with different board states
    print("\n3. Move Predictions:")
    sample_boards = create_sample_board_states()
    
    for board_name, board_state in sample_boards.items():
        print(f"\n   Testing with {board_name} board:")
        try:
            # Test without analysis
            result = client.predict_move(board_state, include_analysis=False)
            action = result['recommended_action']
            print(f"   ✅ Recommended Action: {action['action_name']} (confidence: {action['confidence']:.3f})")
            
            # Test with analysis for one board
            if board_name == "with_holes":
                result_with_analysis = client.predict_move(board_state, include_analysis=True)
                analysis = result_with_analysis['board_analysis']
                print(f"   📊 Board Analysis:")
                print(f"      - Total Holes: {analysis['total_holes']}")
                print(f"      - Max Height: {analysis['max_height']}")
                print(f"      - Bumpiness: {analysis['bumpiness']:.2f}")
                print(f"      - Complete Lines: {analysis['complete_lines']}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ API testing completed!")

def generate_curl_examples():
    """Generate curl command examples for testing."""
    
    print("\n🔧 CURL Command Examples:")
    print("=" * 50)
    
    # Health check
    print("\n1. Health Check:")
    print("curl -X GET http://localhost:8000/health")
    
    # Model info
    print("\n2. Model Information:")
    print("curl -X GET http://localhost:8000/model/info")
    
    # Prediction example
    print("\n3. Move Prediction:")
    sample_board = create_sample_board_states()["empty"]
    payload = {
        "board_state": {
            "board": sample_board
        },
        "include_analysis": True
    }
    
    curl_command = f"""curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(payload, indent=2)}'"""
    
    print(curl_command)
    
    print("\n4. Simplified Prediction (with sample data):")
    simple_payload = {
        "board_state": {
            "board": [[0]*10 for _ in range(22)]  # Empty board
        },
        "include_analysis": False
    }
    
    simple_curl = f"""curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(simple_payload)}'"""
    
    print(simple_curl)

def demonstrate_error_handling():
    """Demonstrate API error handling."""
    
    print("\n⚠️  Error Handling Examples:")
    print("=" * 50)
    
    client = TetrisAPIClient()
    
    # Test with invalid board dimensions
    print("\n1. Invalid Board Dimensions:")
    try:
        invalid_board = [[0]*5 for _ in range(10)]  # Wrong dimensions
        result = client.predict_move(invalid_board)
    except requests.exceptions.HTTPError as e:
        print(f"   ✅ Caught expected error: {e.response.status_code}")
        print(f"   Error details: {e.response.json()['detail']}")
    
    # Test with invalid cell values
    print("\n2. Invalid Cell Values:")
    try:
        invalid_board = [[0]*10 for _ in range(22)]
        invalid_board[0][0] = 10  # Invalid piece value
        result = client.predict_move(invalid_board)
    except requests.exceptions.HTTPError as e:
        print(f"   ✅ Caught expected error: {e.response.status_code}")
        print(f"   Error details: {e.response.json()['detail']}")

if __name__ == "__main__":
    print("🚀 Starting Tetris RL API Test Suite")
    
    # Test the API endpoints
    test_api_endpoints()
    
    # Generate curl examples
    generate_curl_examples()
    
    # Demonstrate error handling
    demonstrate_error_handling()
    
    print("\n🎯 Test suite completed!")
    print("\nTo start the API server, run:")
    print("uvicorn api.main:app --reload --host 0.0.0.0 --port 8000") 