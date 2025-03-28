import chess
from random import choice
from evaluationFunctions.materialValue import calculate_material_value

# Initial minimax caller. Has the move information
def minimax_ai(board):
    if board.turn:
        best_evaluation = -100000
        best_move = choice(list(board.legal_moves))

        for move in board.legal_moves:
            board.push(move)
            evaluation = minimax(board, 2, -100000, 100000, False)
            board.pop()

            if evaluation > best_evaluation:
                best_evaluation = evaluation
                best_move = move
        
        return best_move
    
    else:
        best_evaluation = 100000
        best_move = choice(list(board.legal_moves))

        for move in board.legal_moves:
            board.push(move)
            evaluation = minimax(board, 2, -100000, 100000, True)
            board.pop()

            if evaluation < best_evaluation:
                best_evaluation = evaluation
                best_move = move
        
        return best_move

# Minimax using basic counting evaluation function
def minimax(board, depth, alpha, beta, is_maximizing_player):
    if depth == 0 or board.is_game_over():
        return calculate_material_value(board)

    if is_maximizing_player:
        max_evaluation = -100000

        for move in board.legal_moves:
            board.push(move)
            evaluation = minimax(board, depth - 1, alpha, beta, False)
            board.pop()

            max_evaluation = max(max_evaluation, evaluation)
            alpha = max(alpha, evaluation)

            if beta <= alpha:
                break
        
        return max_evaluation

    else:
        min_evaluation = 100000

        for move in board.legal_moves:
            board.push(move)
            evaluation = minimax(board, depth - 1, alpha, beta, False)
            board.pop()

            min_evaluation = min(min_evaluation, evaluation)
            beta = min(beta, evaluation)

            if beta <= alpha:
                break
        
        return min_evaluation