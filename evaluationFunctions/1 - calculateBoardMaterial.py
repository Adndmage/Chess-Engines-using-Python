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
	bitboardTensor = np.zeros((8, 8, 12))
	for square in chess.SQUARES:
		piece = board.piece_at(square)
		if piece:
			row = square // 8
			col = square % 8
			plane = piece_map[(piece.piece_type, piece.color)]
			bitboardTensor[row][col][plane] = 1
	
	# multiply by piece value (bitboard multiplied with scalar)
	piece_values = [1, 3, 3, 5, 9, 1000, -1, -3, -3, -5, -9, -1000] 
	
	# sum all piece values for each side
	material = np.sum(bitboardTensor * piece_values, axis=(0, 1))

	# return difference between white and black
	return material
    