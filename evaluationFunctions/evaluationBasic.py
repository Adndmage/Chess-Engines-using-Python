import chess
import numpy as np

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 320,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

PAWN_TABLE = np.array([
      0,  0,  0,  0,  0,  0,  0,  0,
     50, 50, 50, 50, 50, 50, 50, 50,
     10, 10, 20, 30, 30, 20, 10, 10,
      5,  5, 10, 25, 25, 10,  5,  5,
      0,  0,  0, 20, 20,  0,  0,  0,
      5, -5,-10,  0,  0,-10, -5,  5,
      5, 10, 10,-20,-20, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0
])

KNIGHT_TABLE = np.array([
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
])

BISHOP_TABLE = np.array([
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
])

ROOK_TABLE = np.array([
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  8,  8,  5,  0,  0
])

QUEEN_TABLE = np.array([
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,   0,  5,  5,  5,  5,  0, -5,
    0,    0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
])

KING_TABLE = np.array([
    -80, -70, -70, -70, -70, -70, -70, -80, 
    -60, -60, -60, -60, -60, -60, -60, -60, 
    -40, -50, -50, -60, -60, -50, -50, -40, 
    -30, -40, -40, -50, -50, -40, -40, -30, 
    -20, -30, -30, -40, -40, -30, -30, -20, 
    -10, -20, -20, -20, -20, -20, -20, -10, 
    20,  20,  -5,  -5,  -5,  -5,  20,  20, 
    20,  30,  10,   0,   0,  10,  30,  20
])

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE
}

def evaluate_position(board):
    evaluation = 0

    for piece_type in PIECE_VALUES.keys():
        material_value = PIECE_VALUES[piece_type]

        # White base evaluation
        white_piece_squares = board.pieces(piece_type, chess.WHITE)
        evaluation += material_value * len(white_piece_squares)
        evaluation += sum(PIECE_SQUARE_TABLES[piece_type][square] for square in white_piece_squares)

        # Black base evaluation
        black_piece_squares = board.pieces(piece_type, chess.BLACK)
        evaluation -= material_value * len(black_piece_squares)
        evaluation -= sum(PIECE_SQUARE_TABLES[piece_type][chess.square_mirror(square)] for square in black_piece_squares)

        # Additional piece placement evaluation
        evaluation += mobility(board, chess.WHITE) - mobility(board, chess.BLACK)
        evaluation += centralization(board, chess.WHITE) - centralization(board, chess.BLACK)

    return evaluation

# Multiple evaluation functions used from https://github.com/supergi0/Python_Chess_AI
def mobility(board, color):
    score = 0

    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
        # mobility_weight = 1 if piece_type == chess.QUEEN else 2

        piece_squares = board.pieces(piece_type, color)
        score += sum(1 for move in board.legal_moves if move.from_square in piece_squares)

    return score

def centralization(board, color):
    score = 0

    for square in [chess.E4, chess.D4, chess.E5, chess.D5]:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            if piece.piece_type == chess.PAWN:
                score += 20
            else:
                score += 10

    return score

def calculate_material_value(board):
    evaluation = 0

    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]

        if piece.color == chess.WHITE:
            evaluation += value
        else:
            evaluation -= value

    return evaluation

if __name__ == "__main__":
    board = chess.Board("4k3/8/4p3/8/2B5/8/8/4K3 w - - 0 1")
    print(mobility(board, chess.WHITE))
