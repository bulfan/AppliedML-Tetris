from PyQt5.QtWidgets import QMainWindow, QFrame, QDesktopWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QWidget, QComboBox
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal, QRect
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush, QLinearGradient
from env.game_data import BOARD_DATA, Shape
from agents.ai_manager import AI_MANAGER
import numpy as np
TETRIS_AI = None
COLORS = {
    'background': QColor(30, 30, 40),
    'board_bg': QColor(20, 20, 30),
    'grid_line': QColor(60, 60, 80),
    'text': QColor(255, 255, 255),
    'accent': QColor(100, 200, 255),
    'success': QColor(100, 255, 100),
    'warning': QColor(255, 200, 100),
    'danger': QColor(255, 100, 100),
    'pieces': {
        0: QColor(20, 20, 30),        
        1: QColor(255, 100, 100),        
        2: QColor(100, 255, 100),        
        3: QColor(100, 100, 255),        
        4: QColor(255, 255, 100),        
        5: QColor(255, 100, 255),        
        6: QColor(100, 255, 255),        
        7: QColor(255, 150, 100),    }
}
class Tetris(QMainWindow):
    def __init__(self):
        super().__init__()
        self.isStarted = False
        self.isPaused = False
        self.isGameOver = False
        self.nextMove = None
        self.lastShape = Shape.shapeNone
        self.ai_mode = False        
        self.ai_speed = 300        
        self.initUI()
    def initUI(self):
        self.gridSize = 25
        self.speed = 300
        self.timer = QBasicTimer()
        self.setFocusPolicy(Qt.StrongFocus)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.tboard = Board(self, self.gridSize)
        main_layout.addWidget(self.tboard)
        self.sidePanel = SidePanel(self, self.gridSize)
        main_layout.addWidget(self.sidePanel)
        self.statusbar = self.statusBar()
        self.statusbar.setStyleSheet("""
            QStatusBar {
                background-color: rgb(30, 30, 40);
                color: white;
                border-top: 1px solid rgb(60, 60, 80);
                font-size: 12px;
                padding: 5px;
            }
        """)
        self.tboard.msg2Statusbar[str].connect(self.statusbar.showMessage)
        self.setStyleSheet("""
            QMainWindow {
                background-color: rgb(30, 30, 40);
            }
        """)
        self.start()
        self.center()
        self.setWindowTitle(' Tetris RL - Deep Q-Network Agent')
        self.show()
        self.setFixedSize(
            self.tboard.width() + self.sidePanel.width() + 60,
            max(self.tboard.height(), self.sidePanel.height()) + self.statusbar.height() + 60
        )
    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)
    def start(self):
        if self.isPaused:
            return
        self.isStarted = True
        self.isGameOver = False
        self.tboard.score = 0
        self.tboard.lines_cleared = 0
        BOARD_DATA.clear()
        self.updateStatusBar()
        BOARD_DATA.createNewPiece()
        self.timer.start(self.speed, self)
    def updateStatusBar(self):
        """Update status bar with current information"""
        mode = " AI" if self.ai_mode else " Manual"
        agent_name = AI_MANAGER.get_current_agent_name() if self.ai_mode else "Human"
        self.tboard.msg2Statusbar.emit(
            f"{mode} | Agent: {agent_name} | Score: {self.tboard.score} | Lines: {self.tboard.lines_cleared}"
        )
    def toggleAIMode(self):
        """Toggle between AI and manual play"""
        self.ai_mode = not self.ai_mode
        if self.ai_mode:
            self.speed = self.ai_speed
        else:
            self.speed = 300
        if not self.isPaused and self.isStarted:
            self.timer.stop()
            self.timer.start(self.speed, self)
        self.updateStatusBar()
        self.sidePanel.update()
    def switchAgent(self, agent_name):
        """Switch to a different AI agent"""
        if AI_MANAGER.set_agent(agent_name):
            self.updateStatusBar()
            self.sidePanel.update()
    def pause(self):
        if not self.isStarted or self.isGameOver: 
            return
        self.isPaused = not self.isPaused
        if self.isPaused:
            self.timer.stop()
            self.tboard.msg2Statusbar.emit("⏸ PAUSED - Press P to resume")
        else:
            self.timer.start(self.speed, self)
            self.updateStatusBar()
        self.updateWindow()
    def updateWindow(self):
        self.tboard.updateData()
        self.sidePanel.updateData()
        self.update()
    def getAIAction(self):
        """Get action from AI agent"""
        if not self.ai_mode:
            return None
        board_state = np.zeros((BOARD_DATA.height, BOARD_DATA.width))
        for x in range(BOARD_DATA.width):
            for y in range(BOARD_DATA.height):
                board_state[y, x] = BOARD_DATA.getValue(x, y)
        action = AI_MANAGER.get_action(BOARD_DATA, board_state, training=False)
        return action
    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():
            if self.isGameOver:
                self.timer.stop()
                return
            if self.ai_mode:
                action = self.getAIAction()
                if action is not None:
                    self.executeAction(action)
            lines = BOARD_DATA.moveDown()
            if lines > 0:
                self.tboard.lines_cleared += lines
                self.tboard.score += lines * 100            
                self.tboard.score += 1            
            if BOARD_DATA.gameOver():
                self.gameOver()
                return
            self.updateStatusBar()
            self.updateWindow()
        else:
            super(Tetris, self).timerEvent(event)
    def executeAction(self, action):
        """Execute a game action"""
        if action == 0:            
            BOARD_DATA.moveLeft()
        elif action == 1:            
            BOARD_DATA.moveRight()
        elif action == 2:            
            BOARD_DATA.rotateLeft()
        elif action == 3:            
            dropped_lines = BOARD_DATA.dropDown()
            self.tboard.score += dropped_lines * 2    
    def keyPressEvent(self, event):
        if self.isGameOver:
            if event.key() == Qt.Key_R:
                self.restartGame()
            return
        if not self.isStarted or BOARD_DATA.currentShape == Shape.shapeNone:
            super(Tetris, self).keyPressEvent(event)
            return
        key = event.key()
        if key == Qt.Key_P:
            self.pause()
            return
        elif key == Qt.Key_A:            
            self.toggleAIMode()
            return
        if self.isPaused:
            return
        if not self.ai_mode:
            if key == Qt.Key_Left:
                BOARD_DATA.moveLeft()
            elif key == Qt.Key_Right:
                BOARD_DATA.moveRight()
            elif key == Qt.Key_Up:
                BOARD_DATA.rotateLeft()
            elif key == Qt.Key_Space:
                dropped_lines = BOARD_DATA.dropDown()
                self.tboard.score += dropped_lines * 2        
        if key == Qt.Key_R:
            self.restartGame()
        else:
            super(Tetris, self).keyPressEvent(event)
        self.updateWindow()
    def gameOver(self):
        self.isGameOver = True
        self.timer.stop()
        mode = " AI" if self.ai_mode else " Manual"
        agent_name = AI_MANAGER.get_current_agent_name() if self.ai_mode else "Human"
        self.tboard.msg2Statusbar.emit(
            f" GAME OVER! {mode} ({agent_name}) | Final Score: {self.tboard.score} | Lines: {self.tboard.lines_cleared} | Press R to Restart"
        )
        self.updateWindow()
    def restartGame(self):
        self.isPaused = False
        self.isGameOver = False
        self.tboard.score = 0
        self.tboard.lines_cleared = 0
        BOARD_DATA.clear()
        BOARD_DATA.createNewPiece()
        self.timer.start(self.speed, self)
        self.updateWindow()
class SidePanel(QFrame):
    def __init__(self, parent, gridSize):
        super().__init__(parent)
        self.gridSize = gridSize
        self.tetris_window = parent
        self.setFixedSize(gridSize * 10, gridSize * BOARD_DATA.height)
        self.setStyleSheet("""
            QFrame {
                background-color: rgb(20, 20, 30);
                border: 2px solid rgb(60, 60, 80);
                border-radius: 10px;
            }
        """)
        self.initAIControls()
    def initAIControls(self):
        """Initialize AI control widgets"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        self.ai_toggle_btn = QPushButton(" Enable AI")
        self.ai_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(60, 60, 80);
                color: white;
                border: 1px solid rgb(100, 100, 120);
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(80, 80, 100);
            }
            QPushButton:pressed {
                background-color: rgb(40, 40, 60);
            }
        """)
        self.ai_toggle_btn.clicked.connect(self.tetris_window.toggleAIMode)
        layout.addWidget(self.ai_toggle_btn)
        self.agent_combo = QComboBox()
        self.agent_combo.addItems(AI_MANAGER.get_available_agents())
        self.agent_combo.setCurrentText(AI_MANAGER.get_current_agent_name())
        self.agent_combo.setStyleSheet("""
            QComboBox {
                background-color: rgb(60, 60, 80);
                color: white;
                border: 1px solid rgb(100, 100, 120);
                border-radius: 5px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
            }
        """)
        self.agent_combo.currentTextChanged.connect(self.tetris_window.switchAgent)
        layout.addWidget(self.agent_combo)
        layout.addStretch()
    def updateData(self):
        if self.tetris_window.ai_mode:
            self.ai_toggle_btn.setText(" Manual Mode")
        else:
            self.ai_toggle_btn.setText(" Enable AI")
        current_agent = AI_MANAGER.get_current_agent_name()
        if self.agent_combo.currentText() != current_agent:
            self.agent_combo.setCurrentText(current_agent)
        self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), COLORS['board_bg'])
        painter.setPen(COLORS['text'])
        painter.setFont(QFont('Arial', 14, QFont.Bold))
        painter.drawText(10, 100, "NEXT PIECE")
        if BOARD_DATA.nextShape:
            minX, maxX, minY, maxY = BOARD_DATA.nextShape.getBoundingOffsets(0)
            dy = 120
            dx = int((self.width() - (maxX - minX) * self.gridSize) / 2)
            val = BOARD_DATA.nextShape.shape
            for x, y in BOARD_DATA.nextShape.getCoords(0, 0, -minY):
                drawSquare(painter, x * self.gridSize + dx, y * self.gridSize + dy, val, self.gridSize)
        y_offset = 200
        painter.setFont(QFont('Arial', 10, QFont.Bold))
        painter.setPen(COLORS['accent'])
        painter.drawText(10, y_offset, " AI CONTROLS")
        y_offset += 80        
        agent_info = AI_MANAGER.get_agent_info()
        painter.setFont(QFont('Arial', 9))
        painter.setPen(COLORS['text'])
        info_lines = [
            f"Agent: {agent_info['name']}",
            f"Type: {agent_info['type']}",
        ]
        if 'epsilon' in agent_info:
            info_lines.append(f"Exploration: {agent_info['epsilon']:.3f}")
        for i, line in enumerate(info_lines):
            painter.drawText(10, y_offset + i * 15, line)
        y_offset += len(info_lines) * 15 + 20
        painter.setFont(QFont('Arial', 10, QFont.Bold))
        painter.setPen(COLORS['accent'])
        painter.drawText(10, y_offset, " CONTROLS")
        y_offset += 20
        controls = [
            "A : Toggle AI/Manual",
            "← → : Move (Manual)",
            "↑ : Rotate (Manual)", 
            "Space : Drop (Manual)",
            "P : Pause",
            "R : Restart"
        ]
        painter.setFont(QFont('Arial', 8))
        painter.setPen(COLORS['text'])
        for i, text in enumerate(controls):
            painter.drawText(10, y_offset + i * 15, text)
class Board(QFrame):
    msg2Statusbar = pyqtSignal(str)
    speed = 10
    def __init__(self, parent, gridSize):
        super().__init__(parent)
        self.setFixedSize(gridSize * BOARD_DATA.width, gridSize * BOARD_DATA.height)
        self.score = 0
        self.lines_cleared = 0
        self.gridSize = gridSize
        self.tetris_window = parent        
        self.setStyleSheet("""
            QFrame {
                background-color: rgb(20, 20, 30);
                border: 3px solid rgb(60, 60, 80);
                border-radius: 10px;
            }
        """)
        self.initBoard()
    def initBoard(self):
        self.score = 0
        self.lines_cleared = 0
        BOARD_DATA.clear()
    def gameOver(self):
        return BOARD_DATA.gameOver()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), COLORS['board_bg'])
        painter.setPen(QPen(COLORS['grid_line'], 1))
        for x in range(BOARD_DATA.width + 1):
            painter.drawLine(x * self.gridSize, 0, x * self.gridSize, BOARD_DATA.height * self.gridSize)
        for y in range(BOARD_DATA.height + 1):
            painter.drawLine(0, y * self.gridSize, BOARD_DATA.width * self.gridSize, y * self.gridSize)
        for x in range(BOARD_DATA.width):
            for y in range(BOARD_DATA.height):
                val = BOARD_DATA.getValue(x, y)
                if val != 0:
                    drawSquare(painter, x * self.gridSize, y * self.gridSize, val, self.gridSize)
        for x, y in BOARD_DATA.getCurrentShapeCoord():
            val = BOARD_DATA.currentShape.shape
            if 0 <= x < BOARD_DATA.width and 0 <= y < BOARD_DATA.height:
                drawSquare(painter, x * self.gridSize, y * self.gridSize, val, self.gridSize, True)
        if self.tetris_window.isGameOver:
            self.drawGameOverMessage(painter)
    def drawGameOverMessage(self, painter):
        overlay = QColor(0, 0, 0, 180)
        painter.fillRect(self.rect(), overlay)
        painter.setPen(COLORS['danger'])
        painter.setFont(QFont('Arial', 24, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, "GAME OVER")
        painter.setPen(COLORS['text'])
        painter.setFont(QFont('Arial', 16))
        score_rect = QRect(0, self.height()//2 + 40, self.width(), 30)
        painter.drawText(score_rect, Qt.AlignCenter, f"Final Score: {self.score}")
        lines_rect = QRect(0, self.height()//2 + 70, self.width(), 30)
        painter.drawText(lines_rect, Qt.AlignCenter, f"Lines Cleared: {self.lines_cleared}")
        painter.setPen(COLORS['accent'])
        painter.setFont(QFont('Arial', 12))
        restart_rect = QRect(0, self.height()//2 + 110, self.width(), 30)
        painter.drawText(restart_rect, Qt.AlignCenter, "Press R to Restart")
    def updateData(self):
        self.update()
def drawSquare(painter, x, y, val, s, is_current=False):
    """Draw a tetris square with modern styling"""
    color = COLORS['pieces'].get(val, COLORS['pieces'][0])
    gradient = QLinearGradient(x, y, x + s, y + s)
    if is_current:
        gradient.setColorAt(0, color.lighter(150))
        gradient.setColorAt(1, color.darker(120))
    else:
        gradient.setColorAt(0, color.lighter(120))
        gradient.setColorAt(1, color.darker(150))
    painter.setBrush(QBrush(gradient))
    painter.setPen(QPen(color.darker(200), 1))
    painter.drawRoundedRect(x + 1, y + 1, s - 2, s - 2, 2, 2)
    if val != 0:
        painter.setPen(QPen(color.lighter(180), 1))
        painter.drawLine(x + 2, y + 2, x + s - 3, y + 2)        
        painter.drawLine(x + 2, y + 2, x + 2, y + s - 3)