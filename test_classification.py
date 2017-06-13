import json
import os
from archetypes.clustering import ClusterSet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "serialization")
INPUTS_DIR = os.path.join(BASE_DIR, "data")

for path in [OUTPUT_DIR, INPUTS_DIR]:
	if not os.path.exists(path):
		os.mkdir(path)

card_map_path = os.path.join(INPUTS_DIR, "card_map.json")
card_map_data = json.loads(open(card_map_path, "rb").read())
card_map = {v: k for k, v in card_map_data["map"].items()}

cluster_set_path = os.path.join(OUTPUT_DIR, "run_1.json")
cluster_set = ClusterSet.from_file(open(cluster_set_path, "rb"))

card_list = ["EX1_319", "CFM_120", "KAR_089", "OG_241", "EX1_308", "CS2_065"]
player_class = "WARLOCK"

cluster = cluster_set.classify_deck(player_class, card_list, card_map)
print(cluster)
