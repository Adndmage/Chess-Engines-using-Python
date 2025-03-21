import chess
from chessAlgorithms.randomPlayer import random_player

def human_player(board):
	"""Gets a move from a human player."""
	while True:
		#print(f"Legal Moves: \n{board.legal_moves}\n")
		move = input(f"Enter move for {'White' if board.turn else 'Black'} (e.g., e2e4): ").strip()
		try:
			chess_move = chess.Move.from_uci(move)
			if chess_move in board.legal_moves:
				return chess_move
			else:
				print("Illegal move! Try again.")
		except ValueError:
			print("Invalid move format! Use UCI notation (e.g., e2e4).")

def play_game(player_white, player_black):
	"""Plays a chess game with two players."""
	board = chess.Board()

	while not board.is_game_over():
		print(board)

		if board.turn:  # White's turn
			move = player_white(board)
		else:  # Black's turn
			move = player_black(board)

		board.push(move)

	print("Game over!")
	print("Result:", board.result())

# Play a human vs human game
play_game(random_player, random_player)
