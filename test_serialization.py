import json
import os
from archetypes.clustering import ClusterSet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIZ_OUTPUT_DIR = os.path.join(BASE_DIR, "visualization")
SIG_OUTPUT_DIR = os.path.join(BASE_DIR, "signatures")

INPUTS_DIR = os.path.join(BASE_DIR, "data")

for path in [VIZ_OUTPUT_DIR, SIG_OUTPUT_DIR, INPUTS_DIR]:
	if not os.path.exists(path):
		os.mkdir(path)

path = os.path.join(INPUTS_DIR, "standard.json")
input_data = None
if os.path.exists(path):
	input_data = json.loads(open(path, "rb").read())

# Load Current GoldStandard ClusterSet
#first_run = ClusterSet.from_input_data(input_data)
serialized_first_run_path = os.path.join(SIG_OUTPUT_DIR, "first_run_output2.json")
#first_run.save(open(serialized_first_run_path, "wb"))

cluster_set = ClusterSet.from_file(open(serialized_first_run_path, "rb"))

print(cluster_set)

# from_file(open("<PATH TO VERSION WITH LABELS>", "r"))
# Generate New ClusterSet From Latest Redshift Data
# new_cluster_set = create_cluster_set(input_data)
# Diff Newest vs GoldStandard and produce new GoldStandard
#new_gold_standard = new_cluster_set.create_new_from_gold_standard(gold_standard)
# Save new GoldStandard
# new_gold_standard.save(open("PATH_FOR_NEW_GOLD_STANDARD", "w"))
