import chess
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chessAlgorithms.minimax import search, iterative_deepening
# r3k2r/p1pp1pb1/bn2pnp1/3PN3/1pq1P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
# Set up a board position
#fen = "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"
fen = "r3k2r/p1pp1pb1/bn2pnp1/3PN3/1pq1P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
board = chess.Board(fen)

# Testing the search function
print("=== Finding Best Move ===")
start_time = time.time()
# best_move = search(board, 2, -100000, 100000)[1]
best_move = iterative_deepening(board, max_depth=5, time_limit=10)
end_time = time.time()
print(f"Best move: {best_move}")
print(f"Time taken: {end_time - start_time:.4f} seconds\n")