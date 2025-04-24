import os
import json
from collections import defaultdict

def create_histogram(data_folder, with_flipped_dataset=False):
    # Initialize histogram with intervals of width 100
    histogram = defaultdict(int)
    interval_width = 1000

    # Prepopulate the histogram with all possible intervals
    for i in range(-10000, 10000 + 1, interval_width):
        histogram[i] = 0

    # Iterate through all files in the folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                # Iterate through the nested lists and extract evaluations
                for game in data:
                    for position in game:
                        evaluation = position["evaluation"]

                        # Check if the evaluation is out of the expected range
                        if evaluation < -10000 or evaluation > 10000:
                            print("Found unexpected evaluation values:")
                            print(f"File: {filename}, Evaluation: {evaluation}")

                        # Determine the interval for the evaluation
                        interval = (evaluation // interval_width) * interval_width
                        histogram[interval] += 1

                        if with_flipped_dataset:
                            # Process the flipped evaluation
                            flipped_evaluation = -evaluation
                            flipped_interval = (flipped_evaluation // interval_width) * interval_width
                            histogram[flipped_interval] += 1

                        

    # Sort and print the histogram
    for interval in sorted(histogram.keys()):
        print(f"[{interval}, {interval + interval_width - 1}] : {histogram[interval]}")

# Specify the folder containing your JSON files
data_folder = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff"
create_histogram(data_folder,True)