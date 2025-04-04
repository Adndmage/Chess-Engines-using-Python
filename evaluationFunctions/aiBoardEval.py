"""
creates 12x8x8 tensor

network
- takes in a 12x8x8 tensor as input, 
	where each plane represents a piece type (6 for white and 6 for black) 
	and each square is 1 if there is a piece otherwise 0.
- structure:
	12x8x8	input layer
	1		fully conected layer (dot product, no activation function)
	1		output node


"""
import json
import chess
import numpy as np

# Define a mapping from pieces to planes in tensor board (White: 0-5, Black: 6-11)
piece_map = {
	(chess.PAWN, chess.WHITE): 0,
	(chess.KNIGHT, chess.WHITE): 1,
	(chess.BISHOP, chess.WHITE): 2,
	(chess.ROOK, chess.WHITE): 3,
	(chess.QUEEN, chess.WHITE): 4,
	(chess.KING, chess.WHITE): 5,
	(chess.PAWN, chess.BLACK): 6,
	(chess.KNIGHT, chess.BLACK): 7,
	(chess.BISHOP, chess.BLACK): 8,
	(chess.ROOK, chess.BLACK): 9,
	(chess.QUEEN, chess.BLACK): 10,
	(chess.KING, chess.BLACK): 11
}

def calculate_board_material(board):
	# calculate numpy bitboard for each piecetype
	bitboardTensor = np.zeros((12, 8, 8)) # all zeros
	for square in chess.SQUARES:
		piece = board.piece_at(square)
		if piece:
			plane = piece_map[(piece.piece_type, piece.color)]
			row = 7 - (square // 8)
			col = square % 8
			bitboardTensor[plane][row][col] = 1 # set to relevant square in the bitboard tensor to 1 if there is a piece 
	return bitboardTensor



def load_game_state(file_path, game_index=0, position_index=0):
	"""
	Load a single game state from a JSON file.
	:param file_path: Path to the JSON file containing game states.
	:param game_index: Index of the game state to load.
	:return: A chess.Board object representing the game state.
	"""
	with open(file_path, 'r') as file:
		games = json.load(file)
		
	# Access the specific game and position
	game = games[game_index]
	position = game[position_index]
	
	# Extract the FEN string and evaluation
	fen = position['board_fen']
	evaluation = position['evaluation']
	
	# Return the board and evaluation
	return chess.Board(fen), evaluation

# Example usage
if __name__ == "__main__":
	# Path to the JSON file
	json_file_path = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\lichess_games_MW1966.json"
	
	# Load the first position of the first game
	board, evaluation = load_game_state(json_file_path, game_index=0, position_index=0)
	
	# Generate the input tensor
	input_tensor = calculate_board_material(board)
	
	# Print the tensor and evaluation for debugging
	print("Input Tensor:")
	print(input_tensor)
	print("Evaluation:")
	print(evaluation)