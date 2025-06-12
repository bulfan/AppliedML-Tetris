"""
AI Manager for Tetris RL
This module manages multiple AI agents and provides a unified interface
for the game to interact with different AI models.
"""
import numpy as np
import os
from agents.dqn_agent import DQNAgent, RandomAgent, HeuristicAgent
from utils.preprocessing import TetrisPreprocessor
from agents.evaluation_agent import EvaluationAgent
import config


class AIManager:
    """Manages multiple AI agents for Tetris gameplay"""
    def __init__(self):
        self.preprocessor = TetrisPreprocessor()
        self.agents = {}
        self.current_agent = None
        self.current_agent_name = "random"
        self.evalMoves = []  # For evaluation agent moves
        self._init_preprocessor()
        self._init_agents()

    def _init_preprocessor(self):
        """Initialize preprocessor with dummy data"""
        dummy_board = np.zeros((config.BOARD_HEIGHT, config.BOARD_WIDTH))
        dummy_features = [self.preprocessor.extract_features(dummy_board) for _ in range(10)]
        self.preprocessor.fit_transform(dummy_features)

    def _init_agents(self):
        """Initialize all AI agents"""
        feature_size = self.preprocessor.get_feature_size()
        action_size = 4        
        self.agents["random"] = RandomAgent(action_size)
        self.agents["heuristic"] = HeuristicAgent(action_size)
        self.agents["dqn_basic"] = DQNAgent(
            state_size=feature_size,
            action_size=action_size,
            lr=1e-3
        )
        self.agents["dqn_advanced"] = DQNAgent(
            state_size=feature_size,
            action_size=action_size,
            lr=5e-4,
        )
        self.agents["evaluation"] = EvaluationAgent()
        self._load_trained_models()
        self.current_agent = self.agents[self.current_agent_name]

    def _load_trained_models(self):
        """Load any available trained models"""
        model_dir = config.MODEL_SAVE_PATH
        if not os.path.exists(model_dir):
            print("No models directory found. Models will be created during training.")
            return
        basic_model_path = os.path.join(model_dir, "tetris_dqn_basic.pth")
        if os.path.exists(basic_model_path):
            try:
                self.agents["dqn_basic"].load(basic_model_path)
                print(f" Loaded basic DQN model from {basic_model_path}")
            except Exception as e:
                print(f"  Failed to load basic DQN model: {e}")
        advanced_model_path = os.path.join(model_dir, "tetris_dqn_advanced.pth")
        if os.path.exists(advanced_model_path):
            try:
                self.agents["dqn_advanced"].load(advanced_model_path)
                print(f" Loaded advanced DQN model from {advanced_model_path}")
            except Exception as e:
                print(f"  Failed to load advanced DQN model: {e}")
        final_model_path = os.path.join(model_dir, "tetris_dqn_final.pth")
        if os.path.exists(final_model_path):
            try:
                self.agents["dqn_advanced"].load(final_model_path)
                print(f" Loaded final DQN model from {final_model_path}")
            except Exception as e:
                print(f"  Failed to load final DQN model: {e}")
        best_basic_path = os.path.join(model_dir, "best_dqn_basic.pth")
        if os.path.exists(best_basic_path):
            try:
                self.agents["dqn_basic"].load(best_basic_path)
                print(f" Loaded best basic DQN model from {best_basic_path}")
            except Exception as e:
                print(f"  Failed to load best basic DQN model: {e}")
        best_advanced_path = os.path.join(model_dir, "best_dqn_advanced.pth")
        if os.path.exists(best_advanced_path):
            try:
                self.agents["dqn_advanced"].load(best_advanced_path)
                print(f" Loaded best advanced DQN model from {best_advanced_path}")
            except Exception as e:
                print(f"  Failed to load best advanced DQN model: {e}")
        evaluation_model_path = os.path.join(model_dir, "evaluation_agent.pth")
        if os.path.exists(evaluation_model_path):
            try:
                self.agents["evaluation"].load_script(evaluation_model_path)
                print(f" Loaded evaluation agent model from {evaluation_model_path}")
            except Exception as e:
                print(f"  Failed to load evaluation agent model: {e}")
        model_files = []
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if not model_files:
            print(" No trained models found. Starting with untrained models.")

    def get_available_agents(self):
        """Get list of available agent names"""
        return list(self.agents.keys())

    def set_agent(self, agent_name):
        """Switch to a different agent"""
        if agent_name in self.agents:
            self.current_agent_name = agent_name
            self.current_agent = self.agents[agent_name]
            print(f"Switched to agent: {agent_name}")
            return True
        else:
            print(f"Agent '{agent_name}' not found. Available agents: {self.get_available_agents()}")
            return False

    def get_current_agent_name(self):
        """Get the name of the current agent"""
        return self.current_agent_name

    def get_action(self, board, board_state, training=False):
        """Get action from current agent"""
        if self.current_agent is None:
            return np.random.randint(4)        
        if self.current_agent_name.startswith("dqn"):
            features = self.preprocessor.extract_features(board_state)
            state_features = self.preprocessor.transform(features)
            print(self.current_agent.act(state_features, training=training))
            return self.current_agent.act(state_features, training=training)
        if self.current_agent_name == "evaluation":
            if len(self.evalMoves) == 0:
                features = self.preprocessor.extract_features(board_state)
                self.evalMoves.extend(self.current_agent.act(board))
            return self.evalMoves.pop(0) if self.evalMoves else None

    def get_agent_info(self):
        """Get information about the current agent"""
        info = {
            "name": self.current_agent_name,
            "type": "unknown",
            "description": "No description available"
        }
        if self.current_agent == "evaluation":
            info.update({
                "type": "evaluation",
                "description": "Evaluation agent that uses a heuristic approach"
            })
        elif self.current_agent_name == "random":
            info.update({
                "type": "baseline",
                "description": "Random action selection baseline"
            })
        elif self.current_agent_name == "heuristic":
            info.update({
                "type": "heuristic",
                "description": "Simple heuristic agent that prefers dropping pieces"
            })
        elif self.current_agent_name.startswith("dqn"):
            info.update({
                "type": "deep_q_network",
                "description": f"Deep Q-Network agent with feature extraction",
                "epsilon": getattr(self.current_agent, 'epsilon', 0.0),
                "step_count": getattr(self.current_agent, 'step_count', 0)
            })
        return info

    def train_step(self, state, action, reward, next_state, done):
        """Training step for DQN agents"""
        if self.current_agent_name.startswith("dqn"):
            state_features = self.preprocessor.transform(
                self.preprocessor.extract_features(state)
            )
            next_state_features = self.preprocessor.transform(
                self.preprocessor.extract_features(next_state)
            )
            self.current_agent.step(state_features, action, reward, next_state_features, done)

    def save_current_model(self, filepath=None):
        """Save the current model if it's a DQN agent"""
        if self.current_agent_name.startswith("dqn"):
            if filepath is None:
                filepath = os.path.join(config.MODEL_SAVE_PATH, f"tetris_{self.current_agent_name}.pth")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.current_agent.save(filepath)
            print(f"Saved {self.current_agent_name} model to {filepath}")
            return True
        else:
            print(f"Cannot save {self.current_agent_name} agent (not a DQN agent)")
            return False
            
        
AI_MANAGER = AIManager() 