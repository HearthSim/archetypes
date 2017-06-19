import json
import os
import random
from archetypes.clustering import ClusterSet
from hearthstone.deckstrings import parse_deckstring
from hearthstone.enums import CardClass
from hearthstone import cardxml
from collections import defaultdict
import json
from itertools import repeat


db, _ = cardxml.load()
dbf_db = {card.dbf_id: card for card in db.values()}

# TEST_DIR = os.path.join("test", "test_data_2017-06-14")

# labeled_data_path = os.path.join(TEST_DIR, "labeled_data.json")
# labeled_data = json.loads(open(labeled_data_path, "rb").read())

# cluster_set_data_path = os.path.join(TEST_DIR, "unlabeled_clusters.json")
# cluster_set_data = open(cluster_set_data_path)

# input_data_path = os.path.join(TEST_DIR, "cluster_input_data.json")
# input_data = json.loads(open(input_data_path).read())

# card_map_path = os.path.join(TEST_DIR, "card_map.json")
# card_map_data = json.loads(open(card_map_path, "rb").read())
# card_map = {v: k for k, v in card_map_data["map"].items()}

# # cluster_set = ClusterSet.from_file(cluster_set_data)
# cluster_set = ClusterSet.from_input_data(input_data)



# def get_card_id(dbf_id):
# 	return dbf_db[dbf_id].id

# def get_cards(deckstring):
# 	try:
# 		cards, _, _ = parse_deckstring(deckstring)
# 	except:
# 		return None
# 	card_ids = []
# 	for dbfId, count in cards:
# 		for i in range(count):
# 			card_ids.append(get_card_id(dbfId))
# 	#return cardids
# 	return card_ids


# for player_class, archetypes in labeled_data.items():
# 	print("\n--- %s archetypes ---" % player_class)
# 	for archetype, decks in archetypes.items():
# 		if archetype.startswith("Custom "):
# 			continue
# 		print("\n-- %i decks in %s--" % (len(decks), archetype))
# 		assigned_clusters = defaultdict(list)
# 		for deck in decks:
# 			cards = get_cards(deck["deckstring"])
# 			if not cards:
# 				continue
# 			cluster, score = cluster_set.classify_deck(player_class.upper(), cards, card_map)
# 			if cluster:
# 				card_names = [db[cardid].name for cardid in cards]
# 				assigned_clusters[cluster].append((card_names, score))
# 		print("assigned to %i clusters" % len(assigned_clusters.keys()))
# 		for cluster, assigned_decks in assigned_clusters.items():
# 			print("\nCLUSTER ID:%s | (%s, %i obs., %i decks): %i (%s%%)" % (cluster.cluster_id, cluster.prevalence, cluster.observations, cluster._num_decks, len(assigned_decks), 100.0 * len(assigned_decks) / len(decks)))
# 			print("SIGNATURE CORE: %s" % (cluster.core_cards))
# 			print("SIGNATURE TECH: %s" % (cluster.tech_cards))
# 			for deck, score in assigned_decks:
# 				print("\t(%s) %s" % (score, deck))


### Classification analysis:

NUM_RUNS = 1
MIN_CARDS = 1
MAX_CARDS = 30

# output dict
classifications = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

# take number of random cards from a deck
def take_random_cards(deck, card_count):
	cards = []
	for card_id, count in deck["cards"].items():
		for _ in range(count):
			cards.append(card_id)
	for _ in range(card_count):
		index = random.randint(0, len(cards) - 1)
		card = cards[index]
		cards.remove(card)
		yield card

# list of full decks with known player_class and archetype
# [{ cards: {1, 2, 2, 3, 4, ...}, player_class: "DRUID", archetype: "Token druid" }, ...]

data = json.loads(open(os.path.join("data", "standard_with_archetypes.json")).read())
archetype_names = json.loads(open(os.path.join("data", "archetype_names.json")).read())


cluster_set = ClusterSet.from_input_data(data)
# card_map = {v: k for k, v in cluster_set.card_map.items()}
# print(card_map)

deck_counts = defaultdict(int)

for player_class_id, decks in cluster_set.decks.items():
	player_class_name = CardClass(int(player_class_id)).name
	player_class  = str(player_class_id)
	print(player_class_name)
	for deck in decks:
		if not deck["archetype_id"]: #or deck["observations"] < 1000:
			continue
		deck_counts[player_class] += 1
		# iterate over number of cards for classification
		for card_count in range(MIN_CARDS, MAX_CARDS + 1):

			# do several runs to get good result
			for _ in range(NUM_RUNS):

				# take given number of random cards from the deck
				# could also introduce completely random cards here
				cards = list(take_random_cards(deck, card_count))

				# classify the random cards
				cluster, score = cluster_set.classify_deck(player_class_name, cards)

				# adjust scores if classification was successful
				if cluster is not None:
					if int(cluster.cluster_id) == int(deck["archetype_id"]):
						classifications[player_class][card_count]["success_count"] += 1
						classifications[player_class][card_count]["success_score"] += score
					else:
						classifications[player_class][card_count]["error_count"] += 1
						classifications[player_class][card_count]["error_score"] += score


# calculate final scores
for player_class_id, data in classifications.items():
	player_class = CardClass(int(player_class_id)).name
	num_decks = deck_counts[player_class_id]
	print("\n-- %s (%s decks) --" % (player_class, num_decks))
	for card_count, values in data.items():
		# average score on successful classification

		success_average_score = values["success_score"] / values["success_count"] if values["success_count"] > 0 else 0
		error_average_score = values["error_score"] / values["error_count"] if values["error_count"] > 0 else 0

		# percent of correctly classified edcks
		total_count = num_decks * NUM_RUNS
		success_percent_classified = 100.0 * values["success_count"] / total_count
		error_percent_classified = 100.0 * values["error_count"] / total_count

		# print("%s correct" % values["success_count"])

		print(
			'%s cards: %s%% success (%s), %s%% error (%s)' %
			(
				card_count, success_percent_classified, success_average_score,
				error_percent_classified, error_average_score
			)
		)

for player_class, clusters in cluster_set.player_class_clusters.items():
	print("\n\n=== %s ===" % player_class)
	print("Common cards: %s" % clusters.clusters[0].common_cards)
	for cluster in clusters.clusters:
		# print(cluster.cluster_id, cluster.observations, len(cluster._decks), len(cluster.cards["core"]), len(cluster.cards["tech"]), len(cluster.cards["common"]))
		print("\nArchetype: %s" % archetype_names[cluster.cluster_id])
		print("Decks: %s, Observations: %s" % (len(cluster._decks), cluster.observations))
		print("Core cards: %s" % cluster.pretty_core_cards)
		print("Tech cards: %s" % cluster.pretty_tech_cards)



# OUTPUT:
# 3 cards: 5% success, score: 5.42
# 4 cards: 17% success, score 6.01
# 5 cards: 36% success, score 14.23
# 6 cards: 42% success, score 17.55
# ...



