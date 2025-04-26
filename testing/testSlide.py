import chess

board = chess.Board("r1bqkbnr/ppp1pQpp/n7/3pN3/3P4/8/PPP1PPPP/RNB1KB1R b KQkq - 0 7")
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
print(f"Turn: {board.turn == chess.BLACK}")
print(board)
print(board.fen())
