import chess

def reorder_moves(board):
    move_list = list(board.legal_moves)

    # Sorts moves to prioritize captures
    def move_sort_key(move):
        if board.is_capture(move):
            return 0
        return 1

    # Reorders the moves
    sorted_move_list = sorted(move_list, key=move_sort_key)

    return sorted_move_list