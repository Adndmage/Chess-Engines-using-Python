"""
minimax implementation to decide upon moves
"""
import chess
from random import choice
from chessAlgorithms.moveOrdering import reorder_moves
from evaluationFunctions.materialValue import calculate_material_value

search_count = 0

def search(board):
    global search_count
    search_count = 0

    best_evaluation = -100000
    best_move = choice(list(board.legal_moves))

    for move in board.legal_moves:
        board.push(move)
        evaluation = -minimax(board, -100000, 100000, 4)
        board.pop()

        if evaluation > best_evaluation:
            best_evaluation = evaluation
            best_move = move
    print(f"Search Count: {search_count}")
    return best_move

def minimax(board, alpha, beta, depth):
    global search_count
    search_count += 1

    if depth == 0:
        return calculate_material_value(board)
    
    best_evaluation = -100000

    for move in board.legal_moves:
        search_count += 1
        board.push(move)
        evaluation = -minimax(board, -alpha, -beta, depth - 1)
        board.pop()

        if evaluation > best_evaluation:
            best_evaluation = evaluation
            alpha = max(alpha, evaluation)
        
        if evaluation >= beta:
            return best_evaluation
    
    return best_evaluation

# def quiescence_search(board, alpha, beta):
#     evaluation = calculate_material_value(board)
#     best_evaluation = evaluation

#     if evaluation >= beta:
#         return evaluation
    
#     if alpha < evaluation:
#         alpha = evaluation
    
    