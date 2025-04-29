import json
import os

def lower_evaluation_precision(data):
    """
    Takes in a list of games (each game is a list of {fen_board, evaluation} dictionaries)
    and rewrites the evaluation values to have only 6 decimal places.
    
    :param data: List of games, where each game is a list of dictionaries with 'fen_board' and 'evaluation'.
    :return: The updated data with evaluations rounded to 6 decimal places.
    """
    for game in data:
        for position in game:
            position['evaluation'] = round(position['evaluation'], 6)
    return data

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define input and output file paths relative to the script's directory
    input_file = os.path.join(current_dir, "newData", "flattened_lichessDatabaseGames-2013-10.json")  # Input file in the newData folder
    output_file = os.path.join(current_dir, "newData", "lessPrecise_flattened_lichessDatabaseGames-2013-10.json")  # Output file in the newData folder

    # Load the JSON file
    with open(input_file, "r") as file:
        data = json.load(file)

    # Lower the precision of evaluation values
    updated_data = lower_evaluation_precision(data)

    # Save the modified data to a new JSON file
    with open(output_file, "w") as file:
        json.dump(updated_data, file, indent=4)

    print(f"Data with lowered precision has been saved to {output_file}.")