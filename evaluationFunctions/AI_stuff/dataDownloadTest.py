import requests
import json
import chess
import chess.pgn

def fetch_lichess_games(username="magnuscarlsen", num_games=10):
    url = f"https://lichess.org/api/games/user/{username}?max={num_games}&evals=true"
    headers = {"Accept": "application/x-ndjson"}

    response = requests.get(url, headers=headers)
    games = response.text.splitlines()  # Each game is a separate line (NDJSON format)
    
    print(f"Total games fetched: {len(games)}")
    #print(f"First game data: {games[0]}")  # Print the first game data for inspection

    data = []
    for game_str in games:
        game = json.loads(game_str)  # Convert string to JSON
        
        # Filter only standard games
        if game.get("variant") != "standard":
            continue
        if game.get("analysis") is None:
            continue

        # print(f"Game data: {game}")  # Print game data for inspection
        
        moves = game.get("moves", "").split()
        evaluations = game.get("analysis", [])
        
        board = chess.Board()
        game_data = []
        
        for i, move in enumerate(moves):
            try:
                board.push_san(move)  # Apply the move to the board
                
                # Get the evaluation if it exists
                eval_data = evaluations[i] if i < len(evaluations) else None # json/dictionary

                if eval_data is None:
                    # Check if the move results in a check
                    if board.is_checkmate():
                        eval_value = -10000 if board.turn else 10000  # Negative for losing, positive for winning
                    else:
                        continue
                
                #print(eval_data)  # Print evaluation data for inspection
                
                if "eval" in eval_data:
                    eval_value = eval_data["eval"]
                elif "mate" in eval_data:
                    print(eval_data["mate"])
                    if eval_data["mate"] > 0:
                        eval_value = 10000 - eval_data["mate"] * 100
                    else:
                        eval_value = -10000 + eval_data["mate"] * 100
                else:
                    # If no "eval" or "mate", check for a check move
                    if board.is_check():
                        eval_value = 10000
                    else:
                        continue
                
                # Add board state and evaluation to the list
                game_data.append({
                    "board_fen": board.fen(),
                    "evaluation": eval_value
                })
            except Exception as e:
                print(f"Error processing move {move}: {e}")
                break
        
        data.append(game_data)
    
    return data

def save_to_file(data, filename="lichess_games.json"):
    """Save the extracted data to a JSON file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving data to file: {e}")


lichess_data = fetch_lichess_games("skiddol", 100)
save_to_file(lichess_data, "lichess_games_skiddol.json")