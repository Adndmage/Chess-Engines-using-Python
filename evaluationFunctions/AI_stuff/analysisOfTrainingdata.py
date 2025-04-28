import os
import json
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP

def create_histogram(data_folder, with_flipped_dataset=False):
    # Initialize histogram with intervals of width 0.01
    histogram = defaultdict(int)
    interval_width = Decimal("0.01")

    # Prepopulate the histogram with all possible intervals
    current = Decimal("-1")
    while current <= Decimal("1"):
        histogram[float(current)] = 0
        current += interval_width

    # Iterate through all files in the folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                # Iterate through the nested lists and extract evaluations
                for game in data:
                    for position in game:
                        evaluation = Decimal(str(position["evaluation"]))

                        # Check if the evaluation is out of the expected range
                        if evaluation < Decimal("-1.1") or evaluation > Decimal("1.1"):
                            print("Found unexpected evaluation values:")
                            print(f"File: {filename}, Evaluation: {evaluation}")

                        # Determine the interval for the evaluation
                        interval = (evaluation / interval_width).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * interval_width
                        histogram[float(interval)] += 1

                        if with_flipped_dataset:
                            # Process the flipped evaluation
                            flipped_evaluation = -evaluation
                            flipped_interval = (flipped_evaluation / interval_width).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * interval_width
                            histogram[float(flipped_interval)] += 1

    # Sort and print the histogram
    for interval in sorted(histogram.keys()):
        print(f"[{interval}, {round(interval + float(interval_width), 2)}] : {histogram[interval]}")

# Specify the folder containing your JSON files
data_folder = r"c:\Users\sebas\Desktop\programmering\DDU\EksamensProjekt DDU\chess bot - eksamensprojekt med leo\evaluationFunctions\AI_stuff\dataInspect"
create_histogram(data_folder, True)