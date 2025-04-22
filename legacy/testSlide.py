import chess

board = chess.Board()
print(board)
print(f"Turn: {board.turn}")

print(f"Legal Moves: \n{board.legal_moves}\n")

print(f"Move legality is {board.is_legal(chess.Move.from_uci('a8a1'))}")
print(chess.Move.from_uci("a8a1") in board.legal_moves)

board.push_san("e4")
board.push_san("e5")
board.push_san("Qh5")
board.push_san("Nc6")
board.push_san("Bc4")
board.push_san("Nf6")
board.push_san("Qxf7")

print(f"Checkmate is {board.is_checkmate()}")
print(board)
print(board.fen())
