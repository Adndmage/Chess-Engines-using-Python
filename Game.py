import chess
from random import shuffle

### import evaluation functions
from evaluationFunctions.calculateBoardMaterial import calculate_board_material
from evaluationFunctions.calculateAIEvalf import evaluate_board

### import chess algorithms
from chessAlgorithms.humanPlayer import human_player
from chessAlgorithms.randomPlayer import random_player
from chessAlgorithms.bestNextMove import bestNextMovePlayer
from chessAlgorithms.minimax import search


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
		move = bestNextMovePlayer(self.board, evaluate_board, side)
		self.board.push(move)
		print(evaluate_board(self.board))
	
	def computer_move_minimax(self):
		move = search(self.board)
		print(move)
		self.board.push(move)