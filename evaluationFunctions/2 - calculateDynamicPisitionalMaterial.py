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


# units value at position
Unit_Value = np.array([[[0, 0, 0, 0, 0, 0, 0, 0], # white pawn
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]]
						,
					   [[-50, -40, -30, -30, -30, -30, -40, -50], # white knight
						[-40, -20,   0,   0,   0,   0, -20, -30],
						[-30,   0,  10,  15,  15,  10,   0, -30],
						[-30,   5,  15,  20,  20,  15,   5, -30],
						[-30,   0,  15,  20,  20,  15,   0, -30],
						[-30,   5,  10,  15,  15,  10,   5, -30],
						[-40, -20,   0,   5,   5,  10, -20, -30],
						[-50, -40, -30, -30, -30, -30, -40, -50]]
						,
					   [[0, 0, 0, 0, 0, 0, 0, 0], # white bishop
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]]
						,
					   [[0, 0, 0, 0, 0, 0, 0, 0], # white rook
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]]
						,
					   [[0, 0, 0, 0, 0, 0, 0, 0], # white queen
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]]
						,
					   [[0, 0, 0, 0, 0, 0, 0, 0], # white king
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]]
						,
					   [[0, 0, 0, 0, 0, 0, 0, 0], # black pawn
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]]
						,
					   [[0, 0, 0, 0, 0, 0, 0, 0], # black knight
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]]
						,
					   [[0, 0, 0, 0, 0, 0, 0, 0], # black bishop
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]]
						,
					   [[0, 0, 0, 0, 0, 0, 0, 0], # black rook
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]]
						,
					   [[0, 0, 0, 0, 0, 0, 0, 0], # white queen
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]]
						,
					   [[0, 0, 0, 0, 0, 0, 0, 0], # white king
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0]]
                        ])


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
	scaling_tensor = np.array([100, 300, 320, 500, 900, 100000, -100, -300, -320, -500, -900, -100000])[:, np.newaxis, np.newaxis] + Unit_Value

	
	# sum all piece values for each side
	square_material = bitboardTensor * scaling_tensor

	# return difference between white and black
	return np.sum(square_material)

	
	# sum all piece values for each side
	# return difference between white and black