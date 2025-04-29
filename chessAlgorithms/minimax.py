"""
minimax implementation to decide upon moves
"""
import time
from math import inf
from chessAlgorithms.moveOrdering import reorder_moves
from evaluationFunctions.evaluationBasic import evaluate_position, calculate_material_value
from evaluationFunctions.allAIEvalInOne import evaluate_board_eval, evaluate_board_10_10_10_eval, evaluate_board_64_32_16_eval, evaluate_board_64_64_32_32_16_16_eval, evaluate_board_128_64_32_16_eval, evaluate_newdata_board_128_64_32_16_eval, evaluate_newdata_board2_128_64_32_16_eval
# from evaluationFunctions.calculateAIEvalf import evaluate_board_1
# from evaluationFunctions.calculateBigAIEvalf import evaluate_board_2
# from evaluationFunctions.calculateBestPerformingAIEval import evaluate_board_3
# from evaluationFunctions.calculateNewBestPerformingAIEval import evaluate_board_4
# from evaluationFunctions.calculateNewInputeTensorAiEvalf import evaluate_board_5

def iterative_deepening(board, max_depth, time_limit=None, engine_type=1):
    start_time = time.time() if time_limit is not None else None

    best_move = None

    for depth in range(1, max_depth + 1):
        # print(f"Depth: {depth}")

        if time_limit and (time.time() - start_time) > time_limit:
            break

        search_result = search(board, depth, -inf, inf, start_time, time_limit, engine_type, preffered_move=best_move)

        if search_result[0] is None:
            break

        move = search_result[1]

        if move is not None:
            best_move = move
    
    # If no move was found, return the first legal move in the ordered move list
    if best_move is None and board.legal_moves:
        best_move = reorder_moves(board)[0]

    return best_move

def search(board, depth, alpha, beta, start_time=None, time_limit=None, engine_type=1, preffered_move=None):
    if start_time is not None and time_limit is not None:
        if time.time() - start_time > time_limit:
            return None, None  # Signal: timeout, no evaluation
    
    if board.is_game_over():
        if board.is_checkmate():
            return -100000, None # Checkmate
        else:
            return 0, None  # Draw
        
    if depth == 0:
        evaluation = quiescence_search(board, alpha, beta, engine_type)
        return evaluation, None
    
    best_evaluation = -inf
    best_move = None

    moves_ordered = reorder_moves(board)

    if preffered_move is not None and preffered_move in moves_ordered:
        moves_ordered.remove(preffered_move)
        moves_ordered.insert(0, preffered_move)

    for move in moves_ordered:
        # Before making a move, check the time limit
        if start_time is not None and time_limit is not None:
            if time.time() - start_time > time_limit:
                return None, None  # Exit early if time limit is exceeded
        
        # Evaluate the move
        board.push(move)
        evaluation = -search(board, depth - 1, -beta, -alpha, engine_type)[0]
        board.pop()

        # Beta-cutoff
        if evaluation >= beta:
            return evaluation, move
        
        if evaluation > best_evaluation:
            best_evaluation = evaluation
            best_move = move

        alpha = max(alpha, evaluation)

        if alpha >= beta:
            break

    return best_evaluation, best_move

def quiescence_search(board, alpha, beta, engine_type=1):
    if board.is_game_over():
        if board.is_checkmate():
            return -100000 # Checkmate
        else:
            return 0  # Draw

    # Get static evaluation relative to WHITE
    evaluation = None
    if engine_type == 1:
        evaluation = evaluate_position(board)
    elif engine_type == 2:
        evaluation = calculate_material_value(board)
    elif engine_type == 3:
        evaluation = evaluate_board_eval(board)
    elif engine_type == 4:
        evaluation = evaluate_board_10_10_10_eval(board)
    elif engine_type == 5:
        evaluation = evaluate_board_64_32_16_eval(board)
    elif engine_type == 6:
        evaluation = evaluate_board_64_64_32_32_16_16_eval(board)
    elif engine_type == 7:
        evaluation = evaluate_board_128_64_32_16_eval(board)
    elif engine_type == 8:
        evaluation = evaluate_newdata_board_128_64_32_16_eval(board)
    elif engine_type == 9:
        evaluation = evaluate_newdata_board2_128_64_32_16_eval(board)
    elif engine_type == 10:
        evaluation = evaluate_board_eval(board) + calculate_material_value(board)
    elif engine_type == 11:
        evaluation = evaluate_board_10_10_10_eval(board) + calculate_material_value(board)
    elif engine_type == 12:
        evaluation = evaluate_board_64_32_16_eval(board) + calculate_material_value(board)
    elif engine_type == 13:
        evaluation = evaluate_board_64_64_32_32_16_16_eval(board) + calculate_material_value(board)
    elif engine_type == 14:
        evaluation = evaluate_board_128_64_32_16_eval(board) + calculate_material_value(board)
    elif engine_type == 15:
        evaluation = evaluate_newdata_board_128_64_32_16_eval(board) + calculate_material_value(board)
    elif engine_type == 16:
        evaluation = evaluate_newdata_board2_128_64_32_16_eval(board) + calculate_material_value(board)
    else:
        raise ValueError("Invalid engine type.")

    stand_pat_white_relative = evaluation
    
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