import chess
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chessAlgorithms.minimax import search

# Set up a board position
fen = "r3k2r/p1pp1pb1/bn2pnp1/3PN3/1pq1P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
board = chess.Board(fen)

# With move reordering
print("=== With Move Reordering ===")
start_time = time.time()
best_move = search(board)  # This uses reorder_moves internally
end_time = time.time()
print(f"Best move: {best_move}")
print(f"Time taken: {end_time - start_time:.4f} seconds\n")