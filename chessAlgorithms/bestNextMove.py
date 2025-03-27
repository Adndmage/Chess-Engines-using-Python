def bestNextMovePlayer(board, evaluationFunction):
    bestMove = None
    bestEvalf = float('-inf')
    for move in board.legal_moves:
        # evaluation function of board after move
        evalf = evaluationFunction(board.push(move))
        #undo move
        board.pop()
        if evalf > bestEvalf:
            bestEvalf = evalf
            bestMove = move
    return bestMove