import chess
import chess.engine

# Path to Stockfish executable
engine_path = r"C:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\stockfish\stockfish-windows-x86-64-avx2.exe"

# Start the Stockfish engine
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# Create a chess board
board = chess.Board()

# Get Stockfish's evaluation of the current position
info = engine.analyse(board, chess.engine.Limit(depth=20))
print("Evaluation:", info["score"].white())

# Quit the engine
engine.quit()