import chess
from chessAlgorithms.randomPlayer import random_player
from chessAlgorithms.bestNextMove import bestNextMovePlayer
from evaluationFunctions.calculateBoardMaterial import calculate_board_material

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
	print(board)

	i = 0
	while not board.is_game_over() and i < 500:
		if board.turn:  # White's turn
			move = player_white(board)
			# move = human_player(board)
		else:  # Black's turn
			move = player_black(board,calculate_board_material, -1)

		board.push(move)

		i += 1
		print(f"\nMove {i}:")
		print(board)

	print("Game over!")
	print("Result:", board.result())

# Play a human vs human game
play_game(random_player, bestNextMovePlayer)