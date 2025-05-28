from env.tetris_env import TetrisEnv
from game_data   import BOARD_DATA, Shape

# 1) Make the env (this calls reset once, which clears the board)
env = TetrisEnv()

# 2) Overwrite the board so the bottom row is completely filled
BOARD_DATA.backBoard = [0] * (env.width * env.height)
for x in range(env.width):
    BOARD_DATA.backBoard[x + (env.height - 1) * env.width] = 1

# 3) Force an O‐shape (so it won’t collide when spawning)
BOARD_DATA.nextShape = Shape(Shape.shapeO)
BOARD_DATA.createNewPiece()

# 4) Hard‐drop via your wrapper
obs, reward, done, _ = env.step(4)

# 5) You should see your print and reward ≥ 1 + (any shaping)
print("After forced drop — reward:", reward)
