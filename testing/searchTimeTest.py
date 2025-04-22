import chess
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chessAlgorithms.minimax import search, search_count

# Set up a board position
fen = "r3k2r/ppp1pppp/1N6/3pbQ2/1nq1P1n1/3b1N2/PPPP1PPP/R1B1KB1R w KQkq - 0 1"
board = chess.Board(fen)

# With move reordering
print("=== With Move Reordering ===")
start_time = time.time()
best_move = search(board)  # This uses reorder_moves internally
end_time = time.time()
print(f"Best move: {best_move}")
print(f"Search count: {search_count}")
print(f"Time taken: {end_time - start_time:.4f} seconds\n")