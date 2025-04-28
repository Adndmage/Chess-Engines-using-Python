import json
import math
import os
from collections import Counter
import numpy as np

# Define the sigmoid function to rescale values between -1 and 1
def sigmoid_rescale(x):
    return 2 / (1 + math.exp(-x)) - 1

# Flatten the distribution using a histogram-inspired approach
def flatten_distribution(evaluations, bins=100000):
    histogram, bin_edges = np.histogram(evaluations, bins=bins, density=True)
    cdf = np.cumsum(histogram) / np.sum(histogram)  # Cumulative distribution function

    # Map each evaluation to its corresponding CDF value
    def map_to_flat(value):
        bin_index = np.digitize(value, bin_edges) - 1
        bin_index = max(0, min(bin_index, len(cdf) - 1))  # Ensure index is within bounds
        return cdf[bin_index] * 2 - 1  # Scale CDF to range [-1, 1]

    return [map_to_flat(x) for x in evaluations]

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output file paths relative to the script's directory
input_file = os.path.join(current_dir, "newData", "lichessDatabaseGames-2013-10.json")  # Input file in the newData folder
output_file = os.path.join(current_dir, "newData", "flattened_lichessDatabaseGames-2013-10.json")  # Output file in the newData folder

# Load the JSON file
with open(input_file, "r") as file:
    data = json.load(file)

# Collect all evaluation scores
all_evaluations = [move["evaluation"] for game in data for move in game]

# Flatten the distribution of evaluation scores
flattened_evaluations = flatten_distribution(all_evaluations)

# Replace the evaluation scores in the original data with the flattened scores
index = 0
for game in data:
    for move in game:
        move["evaluation"] = flattened_evaluations[index]
        index += 1

# Save the modified data to a new JSON file
with open(output_file, "w") as file:
    json.dump(data, file, indent=4)

print(f"Flattened data has been saved to {output_file}.")