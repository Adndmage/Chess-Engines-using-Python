import chess

PIECE_VALUES = {
    chess.PAWN: 10,
    chess.KNIGHT: 30,
    chess.BISHOP: 35,
    chess.ROOK: 50,
    chess.QUEEN: 90,
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