import chess
from random import shuffle

from chessAlgorithms.randomPlayer import random_player
from chessAlgorithms.bestNextMove import bestNextMovePlayer

from evaluationFunctions.calculateBoardMaterial import calculate_board_material

from minimax import minimax_ai

class Game:
	def __init__(self, players):
		self.board = chess.Board()
		self.players = players
	
	def randomize_starting_player(self):
		shuffle(self.players)
	
	def human_move_prompt(self):
		move = input(f"Enter move for {'White' if self.board.turn else 'Black'} (e.g., Nf3): ").strip()

		if self.board.parse_san(move) in self.board.legal_moves:
			return move
	
	def human_move(self):
		try:
			move = self.human_move_prompt()
			self.board.push_san(move)
		except ValueError:
			print("Invalid move notation! Use standard algebraic notation (SAN).")

	def computer_move_random(self):
		move = random_player(self.board)
		self.board.push(move)
	
	def computer_next_best_move(self, side):
		move = bestNextMovePlayer(self.board, calculate_board_material, side)
		self.board.push(move)
	
	def computer_move_minimax(self):
		move = minimax_ai(self.board)
		self.board.push(move)