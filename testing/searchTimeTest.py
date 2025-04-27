import chess
import time
import sys
import os
from math import inf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chessAlgorithms.minimax import search, iterative_deepening
# r3k2r/p1pp1pb1/bn2pnp1/3PN3/1pq1P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 # Complex position 1
# r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 # Complex position 2
# r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 3 3 # Checkmate white
# rnb1k1nr/pppp1ppp/5q2/2b1p3/2B1P3/N7/PPPP1PPP/R1BQK1NR w KQkq - 4 4 # Checkmate black  
# Set up a board position
# rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2 # Englund gambit
fen = "r3k2r/p1pp1pb1/bn2pnp1/3PN3/1pq1P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
board = chess.Board(fen)

# Testing the search function
print("=== Finding Best Move ===")
start_time = time.time()
# best_move = search(board, 3, -inf, inf)[1]
best_move = iterative_deepening(board, max_depth=3, time_limit=20)
end_time = time.time()
print(f"Best move: {best_move}")
print(f"Time taken: {end_time - start_time:.4f} seconds\n")