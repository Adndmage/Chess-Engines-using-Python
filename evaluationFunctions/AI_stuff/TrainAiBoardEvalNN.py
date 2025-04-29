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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

import os  # For checking if the model file exists

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

def boardToTensor(board):
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



def load_dataset_toupleList(file_path):
    """
    Load all game states from a JSON file.
    :param file_path: Path to the JSON file containing game states.
    :return: A list of (board, evaluation) tuples.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    with open(file_path, 'r') as file:
        games = json.load(file)

    dataset = []
    for game in games:
        for position in game:
            #fen = position['board_fen']
            fen = position['fen_board']
            evaluation = position['evaluation']
            if evaluation is None:
                continue
            # Ensure evaluation is normalized to the range [-1, 1]
            if abs(evaluation) > 1.1:
                evaluation = evaluation / 10000  # Normalize if values are in the range [-10000, 10000]
            dataset.append((chess.Board(fen), evaluation))
    return dataset

# Custom Dataset for batching
class ChessDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        board, evaluation = self.dataset[idx]
        #input_tensor = boardToTensor(board)
        input_tensor = boardToTensor2(board) 
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        evaluation_tensor = torch.tensor([evaluation], dtype=torch.float32)
        return input_tensor, evaluation_tensor

# Enable GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    """
    Train the model using DataLoader for batching and GPU acceleration, with validation.
    :param model: The neural network model.
    :param train_loader: DataLoader for the training dataset.
    :param val_loader: DataLoader for the validation dataset.
    :param criterion: Loss function.
    :param optimizer: Optimizer.
    :param epochs: Number of epochs to train.
    """
    model.to(device)  # Move model to GPU
    losses = []  # Store losses for each epoch

    for epoch in range(epochs):
        try:
            print(f"Epoch {epoch + 1}/{epochs}")
            model.train()  # Set model to training mode
            total_train_loss = 0.0

            # Training loop
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * len(inputs)

            # Validation loop
            model.eval()  # Set model to evaluation mode
            total_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item() * len(inputs)

            # Calculate average losses for the epoch
            avg_train_loss = total_train_loss / len(train_loader.dataset)
            avg_val_loss = total_val_loss / len(val_loader.dataset)
            print(f"Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

            # Store losses
            losses.append((epoch + 1, avg_train_loss, avg_val_loss))

        # Check for user input to exit training
        except KeyboardInterrupt:
            print("\nTraining interrupted. Exiting training loop...")
            break  # Exit the loop but continue with the rest of the code

    # Print losses as CSV
    print("\nEpoch,Train Loss,Validation Loss")
    for epoch, train_loss, val_loss in losses:
        print(f"{epoch},{train_loss:.6f},{val_loss:.6f}")

# Define the neural network
class SimpleChessNet(nn.Module): # 769 parameters
    def __init__(self):
        super(SimpleChessNet, self).__init__()
        # Flatten the 12x8x8 tensor into a single vector (768 features)
        self.flatten = nn.Flatten()
        # Fully connected layer: 768 input features -> 1 output feature
        self.fc = nn.Linear(12 * 8 * 8, 1)

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input tensor
        x = self.fc(x)       # Fully connected layer
        return x

class BiggerChessNet(nn.Module): # 7921 parameters
    def __init__(self):
        super(BiggerChessNet, self).__init__()
        # Flatten the 12x8x8 tensor into a single vector (768 features)
        self.flatten = nn.Flatten()
        # First fully connected layer: 768 input features -> 10 nodes
        self.fc1 = nn.Linear(12 * 8 * 8, 10)
        # Second fully connected layer: 10 nodes -> 10 nodes
        self.fc2 = nn.Linear(10, 10)
        # Output layer: 10 nodes -> 1 output
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input tensor
        x = torch.relu(self.fc1(x))  # Apply ReLU activation after the first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation after the second layer
        x = self.fc3(x)  # Final output layer
        return x

class BiggestChessNet(nn.Module): # 51841 parameters
    def __init__(self):
        super(BiggestChessNet, self).__init__()
        # Flatten the 12x8x8 tensor into a single vector (768 features)
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(12 * 8 * 8, 64)  # First layer: 768 -> 64
        self.fc2 = nn.Linear(64, 32)         # Second layer: 64 -> 32
        self.fc3 = nn.Linear(32, 16)         # Third layer: 32 -> 16
        self.fc4 = nn.Linear(16, 1)          # Output layer: 16 -> 1

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input tensor
        
        # Apply ReLU activation after each layer
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        # Output layer (no activation function for regression)
        x = self.fc4(x)
        return x

class BiggerThanBiggestChessNet(nn.Module): # 109313 parameters
    def __init__(self):
        super(BiggerThanBiggestChessNet, self).__init__()
        # Flatten the 12x8x8 tensor into a single vector (768 features)
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(12 * 8 * 8, 128)  # First layer: 768 -> 128
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

class LongerThanBiggestChessNet(nn.Module): # 57329 parameters
    def __init__(self):
        super(LongerThanBiggestChessNet, self).__init__()
        # Flatten the 12x8x8 tensor into a single vector (768 features)
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(12 * 8 * 8, 64)  # First layer: 768 -> 64
        self.fc2 = nn.Linear(64, 64)         # Second layer: 64 -> 64
        self.fc3 = nn.Linear(64, 32)         # Third layer: 64 -> 32
        self.fc4 = nn.Linear(32, 32)         # Fourth layer: 32 -> 32
        self.fc5 = nn.Linear(32, 16)         # Fifth layer: 32 -> 16
        self.fc6 = nn.Linear(16, 16)         # Sixth layer: 16 -> 16
        self.fc7 = nn.Linear(16, 1)          # Output layer: 16 -> 1

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input tensor
        
        # Apply ReLU activation after each layer
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        
        # Output layer (no activation function for regression)
        x = self.fc7(x)
        return x




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



import random

def test_model_on_samples(model, dataset, criterion, num_samples=10):
    """
    Test the model on a random subset of the dataset and print the results.
    :param model: The trained model.
    :param dataset: A list of (board, evaluation) tuples.
    :param criterion: Loss function.
    :param num_samples: Number of random samples to test on.
    """
    model.to(device)  # Move model to GPU
    model.eval()

    # Randomly select samples from the dataset
    random_samples = random.sample(dataset, num_samples)

    print("\nTesting on Random Samples:")
    print(f"{'Actual':>10} {'Predicted':>10} {'Loss':>10}")
    print("-" * 30)

    with torch.no_grad():  # Disable gradient computation
        for board, actual in random_samples:
            # Convert board to tensor and move to GPU
            #input_tensor = torch.tensor(boardToTensor(board), dtype=torch.float32).unsqueeze(0).to(device)
            input_tensor = torch.tensor(boardToTensor2(board), dtype=torch.float32).unsqueeze(0).to(device)


            # Get the model's prediction
            predicted = model(input_tensor).item()

            # Calculate the loss for this sample
            actual_tensor = torch.tensor([actual], dtype=torch.float32).unsqueeze(1).to(device)
            predicted_tensor = torch.tensor([predicted], dtype=torch.float32).unsqueeze(1).to(device)
            loss = criterion(predicted_tensor, actual_tensor).item()

            # Print the results
            print(f"{actual:>10.6f} {predicted:>10.6f} {loss:>10.6f}")

def augment_dataset_with_mirrored_positions(dataset):
    """
    Generate mirrored positions for the dataset.
    :param dataset: A list of (chess.Board, evaluation) tuples.
    :return: A list of mirrored (chess.Board, evaluation) tuples.
    """
    mirrored_positions = []
    for board, evaluation in dataset:
        # Create the mirrored position
        mirrored_board = board.mirror()  # Flip the board
        mirrored_evaluation = -evaluation  # Reverse the evaluation
        mirrored_positions.append((mirrored_board, mirrored_evaluation))
    
    return mirrored_positions

def get_dataset_chunk(dataset, fraction=0.2):
    """
    Get a random fraction of the dataset.
    :param dataset: The full dataset (list of (board, evaluation) tuples).
    :param fraction: The fraction of the dataset to use (e.g., 0.2 for 20%).
    :return: A random subset of the dataset.
    """
    chunk_size = int(len(dataset) * fraction)
    return random.sample(dataset, chunk_size)

# Example usage
if __name__ == "__main__":
    # Path to the JSON file
    json_file_path1 = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\lichess_games_MW1966.json"
    json_file_path2 = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\lichess_games_luka3916.json"
    json_file_path3 = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\lichess_games_skiddol.json"
    json_file_path4 = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\lichess_games_Truemasterme.json"
    json_file_path5 = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\lichess_games_Vlad_Lazarev79.json"
    json_file_path_stockfish = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\stockfish_training_data.json"

    # second round og data
    #json_file_path6 = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\newData\flattened_lichessDatabaseGames-2013-10.json"
    json_file_path6 = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\newData\lessPrecise_flattened_lichessDatabaseGames-2013-10.json"

    # Path to save/load the model
    # model_file_path = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\simpelModel.pth"
    # biggerModel.pth
    #model_file_path = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\biggerModel.pth"
    # biggestModel.pth
    #model_file_path = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\biggestModel.pth"
    # biggerThanBiggestModel.pth
    #model_file_path = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\biggerThanBiggestModel.pth"
    # longerThanBiggestModel.pth
    #model_file_path = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\longerThanBiggestModel.pth"

    # newBiggerThanBiggestModel.pth
    #model_file_path = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\newBiggerThanBiggestModel.pth"
    # newNewTensorBiggerThanBiggestModel.pth
    model_file_path = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\newNewTensorBiggerThanBiggestModel.pth"

    # Load the dataset
    #dataset = load_dataset_toupleList(json_file_path1)
    #dataset += load_dataset_toupleList(json_file_path2)
    #dataset += load_dataset_toupleList(json_file_path3)
    #dataset += load_dataset_toupleList(json_file_path4)
    #dataset += load_dataset_toupleList(json_file_path5)
    #dataset += load_dataset_toupleList(json_file_path_stockfish)
    
    dataset = load_dataset_toupleList(json_file_path6)

    # Augment the dataset with mirrored positions
    dataset += augment_dataset_with_mirrored_positions(dataset)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    batch_size = 1024  # Reduced batch size to balance GPU utilization and CPU workload
    num_workers = 4  # Reduced number of workers to avoid overwhelming the CPU
    prefetch_factor = 1  # Prefetch 2 batches per worker
    train_loader = DataLoader(
        ChessDataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        #prefetch_factor=prefetch_factor,
        persistent_workers=True
    )
    val_loader = DataLoader(
        ChessDataset(val_dataset),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        #prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

    # Initialize the network and move it to GPU
    #model = SimpleChessNet().to(device)
    #model = BiggerChessNet().to(device)
    #model = BiggestChessNet().to(device)
    #model = BiggerThanBiggestChessNet().to(device)
    #model = LongerThanBiggestChessNet().to(device)
    model = NewTensorBiggerThanBiggestChessNet().to(device)

    # Check if a saved model exists
    if os.path.exists(model_file_path):
        print("Loading saved model parameters...")
        model.load_state_dict(torch.load(model_file_path))
    else:
        print("No saved model found. Starting from scratch.")

    # Define a loss function and optimizer
    #criterion = nn.MSELoss()  # Mean Squared Error for regression
    criterion = nn.SmoothL1Loss()  # Huber Loss
        #Directional Accuracy: It penalizes errors proportionally, ensuring that the model learns the correct direction.
        #Magnitude Sensitivity: It considers the magnitude of errors but reduces the impact of large outliers.

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model with training and validation sets
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)

    # Save the model parameters after training
    print("Saving model parameters...")
    torch.save(model.state_dict(), model_file_path)

    # Test the model on 10 random samples
    test_model_on_samples(model, dataset, criterion, num_samples=10)