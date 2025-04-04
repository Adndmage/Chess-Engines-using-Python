"""
creates 12x8x8 tensor

network
- takes in a 12x8x8 tensor as input, 
	where each plane represents a piece type (6 for white and 6 for black) 
	and each square is 1 if there is a piece otherwise 0.
- structure:
	12x8x8	input layer
	1		fully conected layer (dot product, no activation function)
	1		output node


"""

import chess
import numpy as np

# Define a mapping from pieces to planes in tensor board (White: 0-5, Black: 6-11)
piece_map = {
	(chess.PAWN, chess.WHITE): 0,
	(chess.KNIGHT, chess.WHITE): 1,
	(chess.BISHOP, chess.WHITE): 2,
	(chess.ROOK, chess.WHITE): 3,
	(chess.QUEEN, chess.WHITE): 4,
	(chess.KING, chess.WHITE): 5,
	(chess.PAWN, chess.BLACK): 6,
	(chess.KNIGHT, chess.BLACK): 7,
	(chess.BISHOP, chess.BLACK): 8,
	(chess.ROOK, chess.BLACK): 9,
	(chess.QUEEN, chess.BLACK): 10,
	(chess.KING, chess.BLACK): 11
}

def calculate_board_material(board):
	# calculate numpy bitboard for each piecetype
	bitboardTensor = np.zeros((12, 8, 8)) # all zeros
	for square in chess.SQUARES:
		piece = board.piece_at(square)
		if piece:
			plane = piece_map[(piece.piece_type, piece.color)]
			row = 7 - (square // 8)
			col = square % 8
			bitboardTensor[plane][row][col] = 1 # set to relevant square in the bitboard tensor to 1 if there is a piece 



