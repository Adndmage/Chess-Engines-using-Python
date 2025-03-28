import chess
from random import shuffle


### import evaluation functions
from evaluationFunctions.calculateBoardMaterial import calculate_board_material

### import chess algorithms
from chessAlgorithms.humanPlayer import human_player
from chessAlgorithms.randomPlayer import random_player
from chessAlgorithms.bestNextMove import bestNextMovePlayer
from chessAlgorithms.minimax import minimax_ai

"""
The Game manager which holds players
"""

class Game:
	def __init__(self, players):
		self.board = chess.Board()
		self.players = players
	

	### utilities for the "arena"
	def randomize_starting_player(self):
		shuffle(self.players)
	

	### human player handeling
	def human_move(self):
		move = human_player(self.board)
		self.board.push(move)


	### computer player handeling
	def computer_move_random(self):
		move = random_player(self.board)
		self.board.push(move)
	
	def computer_next_best_move(self, side):
		move = bestNextMovePlayer(self.board, calculate_board_material, side)
		self.board.push(move)
	
	def computer_move_minimax(self):
		move = minimax_ai(self.board)
		self.board.push(move)