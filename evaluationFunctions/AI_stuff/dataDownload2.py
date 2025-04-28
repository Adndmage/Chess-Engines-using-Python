import os
import zstandard as zstd
import chess.pgn
import json
import io

def decompress_zst_to_pgn_stream(zst_file):
    """Decompress a .zst file and return a file-like object for the decompressed data."""
    compressed = open(zst_file, 'rb')  # Open the compressed file
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.stream_reader(compressed)
    return io.TextIOWrapper(decompressed, encoding='utf-8'), compressed  # Return both the decompressed stream and the original file handle

def extract_games_from_pgn_stream(pgn_stream, max_games=None):
    """Extract games from a PGN stream and return them in a list format."""
    data = []
    game_count = 0

    while True:
        game = chess.pgn.read_game(pgn_stream)
        if game is None or (max_games and game_count >= max_games):
            break

        # Skip non-standard chess variants
        if game.headers.get("Variant", "Standard") != "Standard":
            continue

        board = game.board()
        game_data = []
        evaluations_available = False

        for node in game.mainline():
            move = node.move
            board.push(move)

            # Extract evaluation from comments (if available)
            comment = node.comment
            eval_value = None
            if "[%eval" in comment:
                try:
                    eval_value = float(comment.split("[%eval ")[1].split("]")[0])
                    evaluations_available = True
                except (IndexError, ValueError):
                    pass
            elif "[%mate" in comment:
                try:
                    mate_value = int(comment.split("[%mate ")[1].split("]")[0])
                    if mate_value > 0:
                        eval_value = 10000 - mate_value * 100
                    else:
                        eval_value = -10000 + mate_value * 100
                    evaluations_available = True
                except (IndexError, ValueError):
                    pass

            if eval_value is not None:
                game_data.append({
                    "fen_board": board.fen(),
                    "evaluation": eval_value
                })

        # Skip games without evaluations
        if evaluations_available:
            data.append(game_data)
            game_count += 1
            if game_count % 1000 == 0:
                print(f"Processed {game_count} games")

    return data

def save_to_file(data, filename="lichess_games.json"):
    """Save the extracted data to a JSON file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving data to file: {e}")

# Example usage
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
zst_file = os.path.join(current_dir, "lichess_db_standard_rated_2013-10.pgn.zst")  # .zst file in the same directory
output_file = os.path.join(current_dir, "lichessDatabaseGames-2013-10.json")  # Output JSON file in the same directory

# Step 1: Decompress the .zst file and process it as a stream
pgn_stream, compressed_file = decompress_zst_to_pgn_stream(zst_file)  # Get both the stream and the original file handle
try:
    # Step 2: Extract games from the PGN stream
    games_data = extract_games_from_pgn_stream(pgn_stream, max_games=10000)  # Adjust max_games as needed
finally:
    # Ensure both the decompressed stream and the original file are closed
    pgn_stream.close()
    compressed_file.close()

# Step 3: Save the extracted data to a JSON file
save_to_file(games_data, output_file)