import torch
import torch.nn as nn
import chess
import numpy as np
#from .AI_stuff.TrainAiBoardEvalNN import NewTensorBiggerThanBiggestChessNet, boardToTensor2
import os


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

def mobility(board, color):
    score = 0

    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
        # mobility_weight = 1 if piece_type == chess.QUEEN else 2

        piece_squares = board.pieces(piece_type, color)
        score += sum(1 for move in board.legal_moves if move.from_square in piece_squares)

    return score / 121 # Normalize by dividing by the maximum possible score (121)

def centralization(board, color):
    score = 0

    for square in [chess.E4, chess.D4, chess.E5, chess.D5]:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            if piece.piece_type == chess.PAWN:
                score += 2
            else:
                score += 1

    return score/8 # Normalize by dividing by the maximum possible score (8)

def boardToTensor2(board):
    # calculate numpy bitboard for each piecetype
    bitboardTensor = np.zeros((14, 8, 8)) # all zeros

    # plane 1 - 12 for each piece types placement (white and black)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            plane = piece_map[(piece.piece_type, piece.color)]
            row = 7 - (square // 8)
            col = square % 8
            bitboardTensor[plane][row][col] = 1 # set to relevant square in the bitboard tensor to 1 if there is a piece 
    
    # Plane 13: Attacked squares map for white
    # Plane 14: Attacked squares map for black
    for square in chess.SQUARES:
        row = 7 - (square // 8)
        col = square % 8

        # Attack map for white
        bitboardTensor[12][row][col] = sum(
            1 for attacker_square in board.pieces(chess.PAWN, chess.WHITE) |
                                    board.pieces(chess.KNIGHT, chess.WHITE) |
                                    board.pieces(chess.BISHOP, chess.WHITE) |
                                    board.pieces(chess.ROOK, chess.WHITE) |
                                    board.pieces(chess.QUEEN, chess.WHITE) |
                                    board.pieces(chess.KING, chess.WHITE)
            if square in board.attacks(attacker_square)
        )
        # Attack map for black
        bitboardTensor[13][row][col] = sum(
            1 for attacker_square in board.pieces(chess.PAWN, chess.BLACK) |
                                    board.pieces(chess.KNIGHT, chess.BLACK) |
                                    board.pieces(chess.BISHOP, chess.BLACK) |
                                    board.pieces(chess.ROOK, chess.BLACK) |
                                    board.pieces(chess.QUEEN, chess.BLACK) |
                                    board.pieces(chess.KING, chess.BLACK)
            if square in board.attacks(attacker_square)
        )
    
    # Use pawn-restricted squares for global information
    # On board 0 (white)and 6 (black) with pawns for each side
    # On row 0 and row 7 for global info
    # Castling rights
    bitboardTensor[0][0][0] = board.has_kingside_castling_rights(chess.WHITE)
    bitboardTensor[6][0][0] = board.has_kingside_castling_rights(chess.BLACK)

    bitboardTensor[0][0][1] = board.has_queenside_castling_rights(chess.WHITE)
    bitboardTensor[6][0][1] = board.has_queenside_castling_rights(chess.BLACK)

    # Mobility scores
    bitboardTensor[0][0][2] = mobility(board, chess.WHITE)
    bitboardTensor[6][0][2] = mobility(board, chess.BLACK)

    # Centralization scores
    bitboardTensor[0][0][3] = centralization(board, chess.WHITE)
    bitboardTensor[6][0][3] = centralization(board, chess.BLACK)


    # count of piece types
    white_total_pieces = len([piece for piece in board.piece_map().values() if piece.color == chess.WHITE])
    black_total_pieces = len([piece for piece in board.piece_map().values() if piece.color == chess.BLACK])
    bitboardTensor[0][7][0] = white_total_pieces / 16  # Normalize by max pieces
    bitboardTensor[6][7][0] = black_total_pieces / 16  # Normalize by max pieces

    # Count of each piece type (normalized)
    for piece_type, max_count in [(chess.PAWN, 8), (chess.KNIGHT, 2), (chess.BISHOP, 2),
                                  (chess.ROOK, 2), (chess.QUEEN, 1), (chess.KING, 1)]:
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        bitboardTensor[0][7][1 + piece_type - 1] = white_count / max_count  # Normalize
        bitboardTensor[6][7][1 + piece_type - 1] = black_count / max_count  # Normalize
    
    return bitboardTensor


class NewTensorBiggerThanBiggestChessNet(nn.Module): # 125697 parameters
    def __init__(self):
        super(NewTensorBiggerThanBiggestChessNet, self).__init__()
        # Flatten the 12x8x8 tensor into a single vector (768 features)
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(14 * 8 * 8, 128)  # First layer: 768 -> 128
        self.fc2 = nn.Linear(128, 64)         # Second layer: 128 -> 64
        self.fc3 = nn.Linear(64, 32)         # Third layer: 64 -> 32
        self.fc4 = nn.Linear(32, 16)          # Output layer: 32 -> 16
        self.fc5 = nn.Linear(16, 1)          # Output layer: 16 -> 1

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input tensor
        
        # Apply ReLU activation after each layer
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        
        # Output layer (no activation function for regression)
        x = self.fc5(x)
        return x


# Dynamically determine the path to the model file
current_dir = os.path.dirname(__file__)  # Directory of the current script
model_file_path = os.path.join(current_dir, "AI_stuff", "newNewTensorBiggerThanBiggestModel.pth")

def evaluate_board_5(board):
    """
    Evaluate a chess board using the trained model.
    :param board: A chess.Board object.
    :return: The evaluation score predicted by the model.
    """
    # Load the trained model
    model = NewTensorBiggerThanBiggestChessNet()
    model.load_state_dict(torch.load(model_file_path))
    model.eval()  # Set the model to evaluation mode

    # Convert the board to a tensor
    input_tensor = boardToTensor2(board)
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():  # Disable gradient computation
        evaluation = model(input_tensor).item()

    return evaluation*1000

# test
if __name__ == "__main__":
    # Create a sample chess board
    board = chess.Board()

    # Evaluate the board
    evaluation = evaluate_board_5(board)
    print(f"Board Evaluation: {evaluation:.6f}")