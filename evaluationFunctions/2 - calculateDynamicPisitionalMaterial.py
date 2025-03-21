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

"""
# units value at position
PAWN_VALUE = np.array([ [7, 7, 7, 7, 7, 7, 7, 7],
						[6, 6, 6, 6, 6, 6, 6, 6],
						[5, 5, 5, 5, 5, 5, 5, 5],
						[4, 4, 4, 4, 4, 4, 4, 4],
						[3, 3, 3, 3, 3, 3, 3, 3],
						[2, 2, 2, 2, 2, 2, 2, 2],
						[1, 1, 1, 1, 1, 1, 1, 1],
						[0, 0, 0, 0, 0, 0, 0, 0]])
# KNIGHT_VALUE
   """ 

def calculate_dynamic_pisitional_material(board):
	# calculate numpy bitboard for each piecetype
	bitboardTensor = np.zeros((12, 8, 8))
	for square in chess.SQUARES:
		piece = board.piece_at(square)
		if piece:
			plane = piece_map[(piece.piece_type, piece.color)]
			row = 7 - (square // 8)
			col = square % 8
			bitboardTensor[plane][row][col] = 1
			
	# multiply by positional piece value (ie by another numpy bitboard instead of a scalar)
	# multiply by piece value (bitboard multiplied with scalar)
	scaling_tensor = np.random.uniform(0.5, 2.0, (12, 8, 8)) * np.array([1, 3, 3, 5, 9, 1000, -1, -3, -3, -5, -9, -1000])[:, np.newaxis, np.newaxis]

	
	# sum all piece values for each side
	square_material = bitboardTensor * scaling_tensor

	# return difference between white and black
	return np.sum(square_material)

	
	# sum all piece values for each side
	# return difference between white and black