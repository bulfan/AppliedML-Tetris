"""
Demo Script for Multiple AI Agents

This script demonstrates different AI agents playing Tetris automatically.
It cycles through different agents and shows their performance.
"""

import sys
import os
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.game_UI import Tetris
from agents.ai_manager import AI_MANAGER

class AIDemo:
    """Demo class that cycles through different AI agents"""
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.game = Tetris()
        self.current_agent_index = 0
        self.agents = AI_MANAGER.get_available_agents()
        self.games_per_agent = 3  # Number of games each agent plays
        self.current_game = 0
        
        self.switch_timer = QTimer()
        self.switch_timer.timeout.connect(self.switch_to_next_agent)
        
        self.performance_stats = {}
        for agent in self.agents:
            self.performance_stats[agent] = {
                'games': 0,
                'total_score': 0,
                'total_lines': 0,
                'avg_score': 0,
                'avg_lines': 0
            }
    
    def start_demo(self):
        """Start the AI demo"""
        print("🎮 Starting AI Agents Demo")
        print("=" * 50)
        print("Available agents:", self.agents)
        print(f"Each agent will play {self.games_per_agent} games")
        print("=" * 50)
        
        self.game.ai_mode = True
        self.game.ai_speed = 200  # Faster for demo
        self.switch_to_agent(0)
        
        self.game.show()
        return self.app.exec_()
    
    def switch_to_agent(self, agent_index):
        """Switch to a specific agent"""
        if agent_index < len(self.agents):
            agent_name = self.agents[agent_index]
            self.current_agent_index = agent_index
            self.current_game = 0
            
            print(f"\n🤖 Switching to agent: {agent_name}")
            AI_MANAGER.set_agent(agent_name)
            
            self.game.switchAgent(agent_name)
            self.game.updateStatusBar()
            
            self.game.restartGame()

    
    def switch_to_next_agent(self):
        """Switch to the next agent in the list"""
        next_index = (self.current_agent_index + 1) % len(self.agents)
        if next_index == 0:
            self.show_final_results()
        else:
            self.switch_to_agent(next_index)
    
    def on_game_over(self):
        """Called when a game ends"""
        current_agent = self.agents[self.current_agent_index]
        score = self.game.tboard.score
        lines = self.game.tboard.lines_cleared
        
        stats = self.performance_stats[current_agent]
        stats['games'] += 1
        stats['total_score'] += score
        stats['total_lines'] += lines
        stats['avg_score'] = stats['total_score'] / stats['games']
        stats['avg_lines'] = stats['total_lines'] / stats['games']
        
        print(f"  Game {stats['games']}: Score {score}, Lines {lines}")
        
        self.current_game += 1
        
        if self.current_game >= self.games_per_agent:
            print(f"✅ {current_agent} completed {self.games_per_agent} games")
            print(f"   Average Score: {stats['avg_score']:.1f}")
            print(f"   Average Lines: {stats['avg_lines']:.1f}")
            
            QTimer.singleShot(2000, self.switch_to_next_agent)
        else:
            QTimer.singleShot(1000, self.game.restartGame)
    
    def show_final_results(self):
        """Show final performance comparison"""
        print("\n🏆 FINAL RESULTS")
        print("=" * 60)
        print(f"{'Agent':<15} {'Games':<6} {'Avg Score':<10} {'Avg Lines':<10}")
        print("-" * 60)
        
        for agent_name in self.agents:
            stats = self.performance_stats[agent_name]
            if stats['games'] > 0:
                print(f"{agent_name:<15} {stats['games']:<6} {stats['avg_score']:<10.1f} {stats['avg_lines']:<10.1f}")
        
        print("=" * 60)
        
        # Find best performing agent
        best_score_agent = max(self.agents, 
                              key=lambda x: self.performance_stats[x]['avg_score'] 
                              if self.performance_stats[x]['games'] > 0 else 0)
        best_lines_agent = max(self.agents, 
                              key=lambda x: self.performance_stats[x]['avg_lines'] 
                              if self.performance_stats[x]['games'] > 0 else 0)
        
        print(f"🥇 Best Score: {best_score_agent} ({self.performance_stats[best_score_agent]['avg_score']:.1f})")
        print(f"🎯 Best Lines: {best_lines_agent} ({self.performance_stats[best_lines_agent]['avg_lines']:.1f})")
        
        # Continue with manual play
        print("\n🎮 Demo completed! You can now play manually or switch agents using the UI.")
        self.game.ai_mode = False
        self.game.updateStatusBar()

def main():
    """Main demo function"""
    # First, run quick training if no models exist
    model_dir = "models"
    if not os.path.exists(model_dir) or not any(f.endswith('.pth') for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))):
        print("🔧 No trained models found. Running quick training first...")
        from scripts.quick_train import main as quick_train_main
        quick_train_main()
        print("\n" + "="*50)
    
    # Start the demo
    demo = AIDemo()
    
    # Override the game's gameOver method to track performance
    original_game_over = demo.game.gameOver
    def enhanced_game_over():
        original_game_over()
        demo.on_game_over()
    demo.game.gameOver = enhanced_game_over
    
    return demo.start_demo()

if __name__ == "__main__":
    main() 