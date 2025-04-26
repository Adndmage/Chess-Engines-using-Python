"""
minimax implementation to decide upon moves
"""
import chess
from random import choice
from chessAlgorithms.moveOrdering import reorder_moves
from evaluationFunctions.materialValue import calculate_material_value
# from evaluationFunctions.calculateAIEvalf import evaluate_board
# from evaluationFunctions.calculateBigAIEvalf import evaluate_board

def search(board, depth, alpha, beta):
    if depth == 0:
        evaluation = quiescence_search(board, alpha, beta)
        return evaluation, None
    
    if board.is_game_over():
        if board.is_checkmate():
            return (-10000 if board.turn else 10000), None # Checkmate
        else:
            return 0, None  # Draw
    
    best_evaluation = -100000
    best_move = None

    moves_ordered = reorder_moves(board)

    for move in moves_ordered:
        # Evaluate the move
        board.push(move)
        evaluation = -search(board, depth - 1, -beta, -alpha)[0]
        board.pop()

        if evaluation >= beta:
            return evaluation, move
        
        if evaluation > best_evaluation:
            best_evaluation = evaluation
            best_move = move

        alpha = max(alpha, evaluation)

        if alpha >= beta:
            break

    return best_evaluation, best_move

def quiescence_search(board, alpha, beta):
    if board.is_game_over():
        if board.is_checkmate():
            return -10000 if board.turn else 10000 # Checkmate
        else:
            return 0  # Draw

    # Get static evaluation relative to WHITE
    stand_pat_white_relative = calculate_material_value(board)
    # Convert to score relative to the CURRENT player
    stand_pat = stand_pat_white_relative if board.turn else -stand_pat_white_relative

    best_evaluation = stand_pat

    if stand_pat >= beta:
        return stand_pat
    
    alpha = max(alpha, stand_pat)
    
    for move in board.legal_moves:
        if not board.is_capture(move):
            continue

        # Evaluate the move
        board.push(move)
        evaluation = -quiescence_search(board, -beta, -alpha)
        board.pop()

        if evaluation >= beta:
            return evaluation
        
        best_evaluation = max(best_evaluation, evaluation)
        alpha = max(alpha, evaluation)

    return best_evaluation


# search_count = 0

# def search(board):
#     global search_count
#     search_count = 0

#     best_evaluation = -100000
#     best_move = choice(list(board.legal_moves))

#     for move in board.legal_moves:
#         board.push(move)
#         evaluation = -minimax(board, -100000, 100000, 3)
#         board.pop()

#         if evaluation > best_evaluation:
#             best_evaluation = evaluation
#             best_move = move
#     print(f"Search Count: {search_count}")
#     return best_move

# def minimax(board, alpha, beta, depth):
#     global search_count
#     search_count += 1

#     if depth == 0:
#         return quiescence_search(board, alpha, beta)
    
#     best_evaluation = -100000

#     for move in board.legal_moves:
#         search_count += 1
#         board.push(move)
#         evaluation = -minimax(board, -alpha, -beta, depth - 1)
#         board.pop()
        
#         if evaluation > best_evaluation:
#             best_evaluation = evaluation
#             alpha = max(alpha, evaluation)
        
#         if evaluation >= beta:
#             return best_evaluation
    
#     return best_evaluation if board.turn else -best_evaluation

# def quiescence_search(board, alpha, beta):
#     stand_pat = calculate_material_value(board)
#     best_evaluation = stand_pat

#     if stand_pat >= beta:
#         return stand_pat
    
#     if alpha < stand_pat:
#         alpha = stand_pat
    
#     for move in [m for m in board.legal_moves if board.is_capture(m)]:
#         board.push(move)
#         evaluation = -quiescence_search(board, -beta, -alpha)
#         board.pop()

#         if evaluation >= beta:
#             return evaluation
        
#         if evaluation > best_evaluation:
#             best_evaluation = evaluation
#             alpha = max(alpha, evaluation)
        
#     return best_evaluation if board.turn else -best_evaluation