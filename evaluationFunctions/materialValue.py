"""
calculates board material by counting up the pieces and multiplying them with their respective values
"""

import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 350,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

def calculate_material_value(board):
    evaluation = 0

    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]

        if piece.color == chess.WHITE:
            evaluation += value
        else:
            evaluation -= value

    return evaluation