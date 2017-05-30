import json
import os
from archetypes.clustering import get_clusters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")

if not os.path.exists(OUTPUT_DIR):
	os.mkdir(OUTPUT_DIR)

if not os.path.exists(INPUTS_DIR):
	os.mkdir(INPUTS_DIR)


def get_input_data(wild=False):
	input_data = None
	path = os.path.join(INPUTS_DIR, "wild.json" if wild else "standard.json")
	if os.path.exists(path):
		input_data = json.loads(open(path, "rb").read())

	return input_data

wild = False
input_data = get_input_data(wild=wild)

clusters = get_clusters(input_data)
heatmap_data = []


for player_class, cluster_set in clusters.items():
	print(str(cluster_set))
	cluster_set.merge_clusters()
	heatmap_data.append(cluster_set.heatmap_data)
	cluster_set.print_summary()

output_path = os.path.join(OUTPUT_DIR, "wild_heatmap_data.json" if wild else "standard_heatmap_data.json")
with open(output_path, "w") as out:
	out.write(json.dumps(heatmap_data, indent=4))
