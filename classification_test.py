import json
import os
from archetypes.clustering import ClusterSet
from hearthstone.deckstrings import parse_deckstring
from hearthstone import cardxml
from collections import defaultdict
import json
from itertools import repeat


db, _ = cardxml.load()
dbf_db = {card.dbf_id: card for card in db.values()}

TEST_DIR = os.path.join("test", "test_data_2017-06-14")

labeled_data_path = os.path.join(TEST_DIR, "labeled_data.json")
labeled_data = json.loads(open(labeled_data_path, "rb").read())

cluster_set_data_path = os.path.join(TEST_DIR, "unlabeled_clusters.json")
cluster_set_data = open(cluster_set_data_path)

input_data_path = os.path.join(TEST_DIR, "cluster_input_data.json")
input_data = json.loads(open(input_data_path).read())

card_map_path = os.path.join(TEST_DIR, "card_map.json")
card_map_data = json.loads(open(card_map_path, "rb").read())
card_map = {v: k for k, v in card_map_data["map"].items()}

# cluster_set = ClusterSet.from_file(cluster_set_data)
cluster_set = ClusterSet.from_input_data(input_data)



def get_card_id(dbf_id):
	return dbf_db[dbf_id].id

def get_cards(deckstring):
	try:
		cards, _, _ = parse_deckstring(deckstring)
	except:
		return None
	card_ids = []
	for dbfId, count in cards:
		for i in range(count):
			card_ids.append(get_card_id(dbfId))
	#return cardids
	return card_ids


for player_class, archetypes in labeled_data.items():
	print("\n--- %s archetypes ---" % player_class)
	for archetype, decks in archetypes.items():
		if archetype.startswith("Custom "):
			continue
		print("\n-- %i decks in %s--" % (len(decks), archetype))
		assigned_clusters = defaultdict(list)
		for deck in decks:
			cards = get_cards(deck["deckstring"])
			if not cards:
				continue
			cluster, score = cluster_set.classify_deck(player_class.upper(), cards, card_map)
			if cluster:
				card_names = [db[cardid].name for cardid in cards]
				assigned_clusters[cluster].append((card_names, score))
		print("assigned to %i clusters" % len(assigned_clusters.keys()))
		for cluster, assigned_decks in assigned_clusters.items():
			print("\nCLUSTER ID:%s | (%s, %i obs., %i decks): %i (%s%%)" % (cluster.cluster_id, cluster.prevalence, cluster.observations, cluster._num_decks, len(assigned_decks), 100.0 * len(assigned_decks) / len(decks)))
			print("SIGNATURE CORE: %s" % (cluster.core_cards))
			print("SIGNATURE TECH: %s" % (cluster.tech_cards))
			for deck, score in assigned_decks:
				print("\t(%s) %s" % (score, deck))


