"""
calculates board material by counting up the pieces and multiplying them with their respective values
"""

import chess

PIECE_VALUES = {
    chess.PAWN: 10,
    chess.KNIGHT: 30,
    chess.BISHOP: 35,
    chess.ROOK: 50,
    chess.QUEEN: 90,
    chess.KING: 1000
}

def calculate_material_value(board):
    if board.is_game_over():
        if board.outcome == chess.WHITE:
            return 100000
        elif board.outcome == chess.BLACK:
            return -100000
        else:
            return 0

    evaluation = 0

    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]

        if piece.color == chess.WHITE:
            evaluation += value
        else:
            evaluation -= value
    
    if board.turn == False:
        evaluation *= -1

    return evaluation