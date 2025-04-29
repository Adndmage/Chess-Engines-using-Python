import torch
import chess
import numpy as np
from .AI_stuff.TrainAiBoardEvalNN import BiggerThanBiggestChessNet, boardToTensor
import os

# Dynamically determine the path to the model file
current_dir = os.path.dirname(__file__)  # Directory of the current script
model_file_path = os.path.join(current_dir, "AI_stuff", "newBiggerThanBiggestModel.pth")

def evaluate_board_4(board):
    """
    Evaluate a chess board using the trained model.
    :param board: A chess.Board object.
    :return: The evaluation score predicted by the model.
    """
    # Load the trained model
    model = BiggerThanBiggestChessNet()
    model.load_state_dict(torch.load(model_file_path))
    model.eval()  # Set the model to evaluation mode

    # Convert the board to a tensor
    input_tensor = boardToTensor(board)
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
    evaluation = evaluate_board_4(board)
    print(f"Board Evaluation: {evaluation:.6f}")