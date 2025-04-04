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
        
        moves = game.get("moves", "").split()
        evaluations = game.get("analysis", [])
        
        #print("\n")
        #print(f"Moves: {moves}")  # Print moves for inspection
        #print(f"Evaluations: {evaluations}")  # Print evaluations for inspection

        
        board = chess.Board()
        game_data = []
        
        for i, move in enumerate(moves):
            try:
                board.push_san(move)  # Apply the move to the board
                
                # Get the evaluation if it exists
                eval_data = evaluations[i] if i < len(evaluations) else None
                eval_value = eval_data.get("eval") if eval_data and "eval" in eval_data else None
                
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

# Example usage
lichess_data = fetch_lichess_games("MW1966", 50)
print()
for game in lichess_data:
    print()
    print(game)  # Print the first 5 board states and evaluations of each game