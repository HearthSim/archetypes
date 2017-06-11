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


for wild in [True, False]:

	if wild:
		fname = "wild.json"
	else:
		fname = "standard.json"

	input_data = None
	path = os.path.join(INPUTS_DIR, fname)
	if os.path.exists(path):
		input_data = json.loads(open(path, "rb").read())

	clusters = ClusterSet.from_input_data(input_data)
	heatmap_data = []

	signatures = {}
	for player_class, cluster_set in clusters.player_class_clusters.items():
		cluster_signatures = {}
		print(str(cluster_set))
		cluster_set.merge_clusters()
		heatmap_data.append(cluster_set.heatmap_data)
		cluster_set.print_summary()
		print("\n\n")
		signatures[player_class] = cluster_set.serialize(legacy=True)


	#heatmap
	output_path = os.path.join(VIZ_OUTPUT_DIR, fname)
	with open(output_path, "w") as out:
		out.write(json.dumps(heatmap_data, indent=4))

	#signatures
	output_path = os.path.join(SIG_OUTPUT_DIR, fname)
	with open(output_path, "w") as out:
		out.write(json.dumps(signatures, indent=4))

	#signature for common deck
	common_clusters_signatures = {}
	for player_class, clusters in signatures.items():
		common_clusters_signatures[player_class] = {}
		for cluster_id, cluster in clusters.items():
			if cluster['prevalence'] == 'common':
				common_clusters_signatures[player_class][cluster_id] = cluster
	output_path = os.path.join(SIG_OUTPUT_DIR, "common-" + fname)
	with open(output_path, "w") as out:
		out.write(json.dumps(common_clusters_signatures, indent=4))