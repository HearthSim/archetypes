import json
import os
from archetypes.clustering import ClusterSet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "serialization")
INPUTS_DIR = os.path.join(BASE_DIR, "data")

for path in [OUTPUT_DIR, INPUTS_DIR]:
	if not os.path.exists(path):
		os.mkdir(path)

path = os.path.join(INPUTS_DIR, "standard.json")
input_data = json.loads(open(path, "rb").read())

path_first = os.path.join(OUTPUT_DIR, "run_1.json")
path_second = os.path.join(OUTPUT_DIR, "run_2.json")

# Step 1: uncomment next 5 lines, run, add names to output
# print("Generating cluster set from input data...")
# first_run = ClusterSet.from_input_data(input_data)
# print("Serailzing cluster set to", path_first)
# first_run.save(open(path_first, "wb"))
# quit()

# Step 2:
first_annotated = ClusterSet.from_file(open(path_first, "rb"))
second_run = ClusterSet.from_input_data(input_data)

second_updated = first_annotated.update(second_run)
second_updated.save(open(path_second, "wb"))

# from_file(open("<PATH TO VERSION WITH LABELS>", "r"))
# Generate New ClusterSet From Latest Redshift Data
# new_cluster_set = create_cluster_set(input_data)
# Diff Newest vs GoldStandard and produce new GoldStandard
#new_gold_standard = new_cluster_set.create_new_from_gold_standard(gold_standard)
# Save new GoldStandard
# new_gold_standard.save(open("PATH_FOR_NEW_GOLD_STANDARD", "w"))
