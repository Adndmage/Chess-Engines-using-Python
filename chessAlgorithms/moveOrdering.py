import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 320,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

def reorder_moves(board):
    move_list = list(board.legal_moves)

    # Captures sorted by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    captures = [move for move in move_list if board.is_capture(move)]
    captures.sort(key=lambda move: (
        PIECE_VALUES.get(board.piece_at(move.to_square).piece_type, 0) -
        PIECE_VALUES.get(board.piece_at(move.from_square).piece_type, 0) if board.piece_at(move.to_square) else 0
    ), reverse=True)
    
    # Divide captures into good captures and bad/neutral captures
    good_captures = [move for move in captures if (PIECE_VALUES.get(board.piece_at(move.to_square).piece_type, 0) - 
                                                    PIECE_VALUES.get(board.piece_at(move.from_square).piece_type, 0)
                                                    if board.piece_at(move.to_square) else 0) > 0]
    bad_captures = [move for move in captures if (PIECE_VALUES.get(board.piece_at(move.to_square).piece_type, 0) - 
                                                    PIECE_VALUES.get(board.piece_at(move.from_square).piece_type, 0)
                                                    if board.piece_at(move.to_square) else 0) <= 0]
    
    # List of checks
    checks = [move for move in move_list if not board.is_capture(move) and board.gives_check(move)]

    # List of quiet moves
    quiet_moves = [move for move in move_list if not board.is_capture(move) and not board.gives_check(move)]

    sorted_move_list = good_captures + checks + bad_captures + quiet_moves

    return sorted_move_list