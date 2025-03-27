def bestNextMovePlayer(board, evaluationFunction):
	bestMove = None
	bestEvalf = float('-inf')

	for move in board.legal_moves:
		# Create a copy of the board and apply the move
		board_copy = board.copy()
		board_copy.push(move)
		
		# Evaluate the board after the move
		evalf = evaluationFunction(board_copy)

		if evalf > bestEvalf:
			bestEvalf = evalf
			bestMove = move
	return bestMove