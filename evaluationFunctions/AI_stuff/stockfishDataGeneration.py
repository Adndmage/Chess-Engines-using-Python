import chess
import chess.engine
import json
import os

def generate_training_data(board_fen, output_file, stockfish_path, depth=20):
    """
    Generate training data by evaluating a board position and all legal moves from that position.
    :param board_fen: The FEN string of the starting board position.
    :param output_file: Path to the JSON file where data will be appended.
    :param stockfish_path: Path to the Stockfish executable.
    :param depth: Depth for Stockfish evaluation.
    """
    # Start the Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Load the board from the provided FEN
    board = chess.Board(board_fen)

    # List to store evaluations for the current position and its legal moves
    position_data = []

    # Evaluate the current board position
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    evaluation = info["score"].white().score(mate_score=10000)  # Mate in X is ±10000
    position_data.append({
        "board_fen": board.fen(),
        "evaluation": evaluation
    })

    # Evaluate all legal moves from the current position
    for move in board.legal_moves:
        board.push(move)  # Make the move
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        evaluation = info["score"].white().score(mate_score=10000)
        position_data.append({
            "board_fen": board.fen(),
            "evaluation": evaluation
        })
        board.pop()  # Undo the move

    # Append the data to the JSON file
    if os.path.exists(output_file):
        with open(output_file, "r") as file:
            data = json.load(file)
    else:
        data = []

    data.append(position_data)

    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)

    # Quit the engine
    engine.quit()

# Example usage
if __name__ == "__main__":
    # Path to Stockfish executable
    stockfish_path = r"C:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\stockfish\stockfish-windows-x86-64-avx2.exe"

    # Output JSON file
    output_file = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\stockfish_training_data.json"

    # Example board position (starting position)
    board_fen = "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"

    # Generate training data
    generate_training_data(board_fen, output_file, stockfish_path, depth=20)