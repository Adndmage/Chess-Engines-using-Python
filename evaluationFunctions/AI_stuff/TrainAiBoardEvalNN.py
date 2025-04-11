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

def load_dataset_toupleList(file_path):
    """
    Load all game states from a JSON file.
    :param file_path: Path to the JSON file containing game states.
    :return: A list of (board, evaluation) tuples.
    """
    with open(file_path, 'r') as file:
        games = json.load(file)
    
    dataset = []
    for game in games:
        for position in game:
            fen = position['board_fen']
            evaluation = position['evaluation'] 
            if evaluation is None:
                continue
            dataset.append((chess.Board(fen), evaluation/1000)) # to touple and Normalize evaluation
    return dataset

def train_model(model, dataset, criterion, optimizer, epochs=10):
    """
    Train the model on the dataset for a specified number of epochs.
    :param model: The neural network model.
    :param dataset: A list of (board, evaluation) tuples.
    :param criterion: Loss function.
    :param optimizer: Optimizer.
    :param epochs: Number of epochs to train.
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0

        # Shuffle the dataset at the start of each epoch
        np.random.shuffle(dataset)

        for board, evaluation in dataset:
            # Convert board to tensor
            input_tensor = boardToTensor(board)
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            evaluation_tensor = torch.tensor([evaluation], dtype=torch.float32).unsqueeze(1)  # Match output shape

            # Forward pass
            output = model(input_tensor)
            loss = criterion(output, evaluation_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        # Print average loss for the epoch
        print(f"Average Loss: {total_loss / len(dataset):.6f}")


# Define the neural network
class SimpleChessNet(nn.Module):
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

import random

def test_model_on_samples(model, dataset, criterion, num_samples=10):
    """
    Test the model on a random subset of the dataset and print the results.
    :param model: The trained model.
    :param dataset: A list of (board, evaluation) tuples.
    :param criterion: Loss function.
    :param num_samples: Number of random samples to test on.
    """
    # Set the model to evaluation mode
    model.eval()

    # Randomly select samples from the dataset
    random_samples = random.sample(dataset, num_samples)

    print("\nTesting on Random Samples:")
    print(f"{'Actual':>10} {'Predicted':>10} {'Loss':>10}")
    print("-" * 30)

    with torch.no_grad():  # Disable gradient computation
        for board, actual in random_samples:
            # Convert board to tensor
            input_tensor = boardToTensor(board)
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            # Get the model's prediction
            predicted = model(input_tensor).item()

            # Calculate the loss for this sample
            actual_tensor = torch.tensor([actual], dtype=torch.float32).unsqueeze(1)  # Match output shape
            predicted_tensor = torch.tensor([predicted], dtype=torch.float32).unsqueeze(1)
            loss = criterion(predicted_tensor, actual_tensor).item()

            # Print the results
            print(f"{actual:>10.6f} {predicted:>10.6f} {loss:>10.6f}")

# Example usage
if __name__ == "__main__":
    # Path to the JSON file
    json_file_path1 = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\lichess_games_MW1966.json"
    json_file_path2 = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\lichess_games_luka3916.json"
    json_file_path3 = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\lichess_games_skiddol.json"
    json_file_path4 = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\stockfish_training_data.json"

    # Path to save/load the model
    model_file_path = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\model.pth"

    # Load the dataset
    dataset = load_dataset_toupleList(json_file_path1)
    dataset += load_dataset_toupleList(json_file_path2)
    dataset += load_dataset_toupleList(json_file_path3)
    dataset += load_dataset_toupleList(json_file_path4)
    

    # Initialize the network
    model = SimpleChessNet()

    # Check if a saved model exists
    if os.path.exists(model_file_path):
        print("Loading saved model parameters...")
        model.load_state_dict(torch.load(model_file_path))
    else:
        print("No saved model found. Starting from scratch.")

    # Define a loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataset, criterion, optimizer, epochs=10)

    # Save the model parameters after training
    print("Saving model parameters...")
    torch.save(model.state_dict(), model_file_path)

    # Test the model on 10 random samples
    test_model_on_samples(model, dataset, criterion, num_samples=10)