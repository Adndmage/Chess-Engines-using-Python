"""
randomly choce a move from the legal moves
"""
import random
def random_player(board):
	"""Plays a random move."""
	return random.choice(list(board.legal_moves))