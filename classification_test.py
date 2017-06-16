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


### Classification analysis:

NUM_RUNS = 30
MIN_CARDS = 1
MAX_CARDS = 30

# output dict
classifications = defaultdict(defaultdict(float))

# take number of random cards from a deck
def take_random_cards(deck, card_count):
	cards = list(deck.cards)
	for _ in range(card_count):
		index = random.randint(0, len(cards) - 1)
		card = cards[index]
		cards.remove(card)
		yield card

# list of full decks with known player_class and archetype
# [{ cards: {1, 2, 2, 3, 4, ...}, player_class: "DRUID", archetype: "Token druid" }, ...]
classified_decks = []

# cluster set generated from the list above
cluster_set = {}

# iterate over all known decks
for deck in classified_decks:

	# iterate over number of cards for classification
	for card_count in range(MIN_CARDS, MAX_CARDS):

		# do several runs to get good result
		for _ in range (NUM_RUNS):

			# take given number of random cards from the deck
			# could also introduce completely random cards here
			cards = list(take_random_cards(deck, card_count))

			# classify the random cards
			cluster, score = cluster_set.classify_deck(cards)

			# adjust scores if classification was successful
			if cluster.name == deck.archetype:
				classifications[card_count]["count"] += 1
				classifications[card_count]["score"] += score

# calculate final scores
for card_count, values in classifications.items():
	# average score on successful classification
	average_score = values["score"] / values["count"]

	# percent of correctly classified edcks
	percent_classified = values["count"] / len(classified_decks) * NUM_RUNS * (MAX_CARDS - MIN_CARDS)

	print("%i cards: %f%% success, score: %f" % (card_count, 100.0 * percent_classified, average_score))



# OUTPUT:
# 3 cards: 5% success, score: 5.42
# 4 cards: 17% success, score 6.01
# 5 cards: 36% success, score 14.23
# 6 cards: 42% success, score 17.55
# ...



