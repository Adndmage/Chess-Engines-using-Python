"""
Allows the player to input moves in standard algebraic notation (SAN) using the console.
"""
def human_player(board):
	while True:
		move = input(f"Enter move for {'White' if board.turn else 'Black'} (e.g., Nf3): ").strip()
		try:
			chess_move = board.parse_san(move)
			if chess_move in board.legal_moves:
				return chess_move
			else:
				print("Illegal move! Try again.")
		except ValueError:
			print("Invalid move notation! Use standard algebraic notation (SAN).")