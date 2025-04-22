"""
minimax implementation to decide upon moves
"""
import chess
from random import choice
from evaluationFunctions.materialValue import calculate_material_value

def search(board):
    best_evaluation = -100000
    best_move = choice(list(board.legal_moves))

    for move in board.legal_moves:
        board.push(move)
        evaluation = -minimax(board, -100000, 100000, 3)
        board.pop()

        if evaluation > best_evaluation:
            best_evaluation = evaluation
            best_move = move
    
    return best_move

def minimax(board, alpha, beta, depth):
    if depth == 0 or board.is_game_over():
        return calculate_material_value(board)
    
    best_evaluation = -100000

    for move in board.legal_moves:
        board.push(move)
        evaluation = -minimax(board, -alpha, -beta, depth - 1)
        board.pop()

        if evaluation > best_evaluation:
            best_evaluation = evaluation
            alpha = max(alpha, evaluation)
        
        if evaluation >= beta:
            return best_evaluation
    
    return best_evaluation