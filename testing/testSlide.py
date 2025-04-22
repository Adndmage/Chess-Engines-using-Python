import chess

board = chess.Board("3k4/8/8/8/4K3/8/8/8 w - - 0 1")
print(board)

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
print(f"Turn: {board.turn == chess.BLACK}")
print(board)
print(board.fen())
