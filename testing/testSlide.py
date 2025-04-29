import chess

board = chess.Board("1rbqkbnr/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQk - 0 4")
print(board)

print(f"Legal Moves: \n{board.legal_moves}\n")

print(f"Move legality is {board.is_legal(chess.Move.from_uci('a8a1'))}")
print(chess.Move.from_uci("a8a1") in board.legal_moves)

# board.push_san("e4")
# board.push_san("e5")
# board.push_san("Qh5")
# board.push_san("Nc6")
# board.push_san("Bc4")
# board.push_san("Nf6")
# board.push_san("Qxf7")

print(f"Checkmate is {board.is_checkmate()}")
print(f"Outcome: {board.outcome().result() == '1-0' if board.outcome() else None}")
print(f"Turn: {board.turn == chess.BLACK}")
print(board)
print(board.fen())