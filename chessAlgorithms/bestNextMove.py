import random
def bestNextMovePlayer(board, evaluationFunction, side): # side = black = -1, white = 1
	# has a list of best moves such that a random can be chocen in case they are equal
	bestMoves = []
	bestEvalf = -10000

	for move in board.legal_moves:
		# Create a copy of the board and apply the move
		board_copy = board.copy()
		board_copy.push(move)
		
		# Evaluate the board after the move
		evalf = evaluationFunction(board_copy) * side

		if evalf > bestEvalf:
			bestEvalf = evalf
			bestMoves = [move] # replace list of best moves if one is better
		elif evalf == bestEvalf:
			bestMoves.append(move) # add to best move posibilities if equal
		
	return random.choice(bestMoves)