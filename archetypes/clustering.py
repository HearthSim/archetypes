import pprint
import numpy as np
from itertools import combinations
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hearthstone.enums import CardClass
from hearthstone.cardxml import load


pp = pprint.PrettyPrinter(indent=4)
db, _ = load()


def cluster_similarity(c1, c2):
	"Compute a weighted similarity based of the signatures"

	## Maybe use tech cards as well?
	c1_sig = c1.signature['core']
	c1_card_list = c1_sig.keys()
	c2_sig = c2.signature['core']
	c2_card_list = c2_sig.keys()

	intersection = list(set(c1_card_list) & set(c2_card_list))
	union = list(set(c1_card_list) | set(c2_card_list))
	flat_score = float(len(intersection)) / float(len(union))

	# Weighted score which is the same as computing the new signature
	values = {}
	for c in union:
		if c in c1_sig and c in c2_sig:
			values[c] = (c1_sig[c] + c2_sig[c]) / 2
		elif c in c1_sig:
			values[c] = c1_sig[c]
		else:
			values[c] = c2_sig[c]
	w_intersection = 0.0
	for c in intersection:
		w_intersection += values[c]

	w_union = 0.0
	for c in union:
		w_union += values[c]

	weighted_score = float(w_intersection) / float(w_union)
	return weighted_score


class ClusterSet:
	"""A container for a class's deck clusters."""
	def __init__(self, player_class, card_map, decks):
		self.NUM_CLUSTERS_REQUESTED = 10
		# % of deck within a cluster that need to have the card to be considered core card
		self.CORE_CARD_DECK_THRESHOLD = 0.8
		# % of deck within a cluster that need to have a card to be considerd a tech card
		self.TECH_CARD_DECK_THRESHOLD = 0.3
		# % of decks that need to contains the card as to count it as common_card accross the class (remove it from signatures)
		self.COMMON_CARD_CLUSTER_THRESHOLD = 0.93
		self.LOW_VOLUME_CLUSTER_MULTIPLIER = 1.5
		self._low_observation_volume_cutoff = None
		self.clusters = []

		self.player_class = player_class
		self._card_map = card_map

		self._decks = decks
		self._deck_matrix = []
		self.merge_history = {}
		self._merge_pass = 1

		common_cards_tmp = defaultdict(float)
		for deck in self._decks:
			for card, count in deck['cards'].items():
				common_cards_tmp[card] += 1.0

		common_cards = {}
		num_decks = float(len(self._decks))
		for card, count in common_cards_tmp.items():
			prevalence = count / num_decks
			if prevalence >= self.COMMON_CARD_CLUSTER_THRESHOLD:
				common_cards[card] = prevalence

		for deck in self._decks:
			# deck is a dict with 'deck_list', 'cards', 'observations'
			vector = [0] * len(self._card_map)
			for card, count in deck['cards'].items():
				if card in common_cards:
					continue

				vector[int(card)] = count
			self._deck_matrix.append(vector)

		self.common_cards_map = common_cards
		self.common_cards = [self.card_name(c) for c in common_cards.keys()]

		self.make_clusters()

	def __repr__(self):
		return str(self)

	def __str__(self):
		args = (self.cluster_class_name, self.total_deck_count, self.deck_count, self.cluster_count)
		return "%s: %i deck observations of %i distinct lists across %i clusters" % args

	def print_most_common_cards(self):
		for card, score in self.common_cards_map:
			print("{card:>30} -> {score}".format(card = card, score=score))

	@property
	def deck_count(self):
		return len(self._decks)

	@property
	def total_deck_count(self):
		# float(sum(d['observations'] for d in self._decks)) - previous implementation
		return float(sum(1 for d in self._decks))

	@property
	def cluster_class_name(self):
		return self.player_class.name

	@property
	def heatmap_data(self):
		cluster_ids = []
		zValues = []

		card_list = []
		num_clusters = len(self.clusters)
		cards = {}

		cluster_id = 0
		for cluster in self.clusters:
			for ctype, members in cluster.signature.items():
				for card, value in members.items():
					if card not in cards:
						cards[card] = [0] * num_clusters
					if ctype == "core":
						cards[card][cluster_id] = 1
					else:
						cards[card][cluster_id] = .5

			cluster_id  += 1
			cluster_ids.append(cluster_id)

		for card_id, clusters in cards.items():
			card_name = self.card_name(card_id)
			card_list.append(card_name)
			zValues.append(clusters)

		common_cards = [self.card_name(c) for c in self.common_cards_map.keys()]
		card_list.extend(common_cards)
		data = {
			"class": self.cluster_class_name,
			"common_cards": common_cards,
			"num_clusters": len(self.clusters),
			"card_list": card_list,
			"clusters": None,
			"cluster_ids": cluster_ids
		}

		clusters_for_data = []
		for cluster in self.clusters:
			clusters_for_data.append({
				"core_cards": cluster.pretty_core_cards,
				"tech_cards": cluster.pretty_tech_cards,
				"num_decks": cluster.deck_count,
				"num_observations": cluster.observations,
				"prevalence": cluster.prevalence,
				"winrate": cluster.win_rate
			})

		clusters_for_data = sorted(clusters_for_data, key=lambda c: c["num_observations"], reverse=True)
		data["clusters"] = clusters_for_data
		return data

	@property
	def cluster_count(self):
		return float(len(self.clusters))

	def card_name(self, card_index):
		card_id = self._card_map[int(card_index)]
		return db[card_id].name.replace(",", "")

	def make_clusters(self):
		self.clusters = []

		# Then do clustering work
		X = self._deck_matrix
		X = StandardScaler().fit_transform(X)
		self._clusterizer = KMeans(
			n_clusters=min(self.NUM_CLUSTERS_REQUESTED, len(self._deck_matrix))
		)
		self._clusterizer.fit(X)

		decks_per_cluster = defaultdict(list)
		for deck, cluster_id in zip(self._decks, self._clusterizer.labels_):
			decks_per_cluster[cluster_id].append(deck)

		for cluster_id, decks in decks_per_cluster.items():
			c = Cluster(
				self,
				str(cluster_id),
				decks
			)
			self.clusters.append(c)

	@property
	def low_observation_volume_cutoff(self):
		if not self._low_observation_volume_cutoff:
			na = np.array([cluster.observations for cluster in self.clusters])
			avg = np.mean(na, axis=0)
			# std = np.std(na, axis=0)
			# total = np.sum(na, axis=0)
			self._low_observation_volume_cutoff = avg / self.LOW_VOLUME_CLUSTER_MULTIPLIER
		return self._low_observation_volume_cutoff

	def merge_clusters(self, distance_function=cluster_similarity):
		dist, obsv = self.analyze_clusters_space(self.clusters, distance_function)
		self.clusters = self._do_merge_clusters(self.clusters, distance_function, dist, obsv)

	def analyze_clusters_space(self, clusters, distance_function):
		"Determine reasonable parameters for second phase of clustering"
		previous_c1 = None
		distances = []
		distances_all = []
		observations_all = []
		for c1, c2 in combinations(clusters, 2):
			sim_score = distance_function(c1, c2)
			if c1 != previous_c1:
				if len(distances):
					print("%s - %s" % (c1.cluster_id, distances))
				previous_c1 = c1
				distances = []
			distances.append(round(sim_score, 2))
			distances_all.append(round(sim_score, 2))

		wr = np.array(distances_all)
		mean = np.mean(wr, axis=0)
		std = np.std(wr, axis=0)
		max_val = np.max(wr, axis=0)
		min_val = np.min(wr, axis=0)
		distance_threshold = mean + (std * 2) # or 3
		# print "distance tresh: %s, mean:%s, std:%s\n" % (round(distance_threshold, 2), round(mean,2), round(std, 2))

		observations = []
		for cluster in clusters:
			observations.append(sum(d['observations'] for d in cluster._decks))
		wr = np.array(observations)
		mean = np.mean(wr, axis=0)
		std = np.std(wr, axis=0)
		max_val = np.max(wr, axis=0)
		min_val = np.min(wr, axis=0)
		observation_threshold = mean / self.LOW_VOLUME_CLUSTER_MULTIPLIER # or 3
		#print "observations: %s" % observations
		#print "observation tresh: %s, mean:%s, std:%s (can't be above)\n" % (round(observation_threshold, 2), round(mean,2), round(std, 2))

		return distance_threshold, observation_threshold

	def _do_merge_clusters(self, clusters, distance_function, distance_threshold, observation_threshold):
		next_cluster_id = len(self.clusters)
		current_clusters = list(clusters)

		while True:
			most_similar = self._most_similar_pair(current_clusters, distance_function, observation_threshold)
			if not most_similar or most_similar[2] < distance_threshold:
				break
			c1, c2, sim_score = most_similar
			new_cluster_decks = []
			new_cluster_decks.extend(c1._decks)
			new_cluster_decks.extend(c2._decks)
			new_cluster = Cluster(
				self,
				next_cluster_id,
				new_cluster_decks,
				parents=[c1, c2],
				parent_similarity=sim_score
			)
			next_cluster_id += 1
			next_clusters_list = [new_cluster]
			for c in current_clusters:
				if c.cluster_id not in (c1.cluster_id, c2.cluster_id):
					next_clusters_list.append(c)
			current_clusters = next_clusters_list

		return current_clusters

	def _most_similar_pair(self, clusters, distance_function, observation_threshold):
		result = []
		history = []
		cluster_ids = set()
		for c1, c2 in combinations(clusters, 2):
			if c1.observations > observation_threshold and c2.observations > observation_threshold:
				continue
			cluster_ids.add("c%s" % c1.cluster_id)
			cluster_ids.add("c%s" % c2.cluster_id)

			sim_score = distance_function(c1, c2)
			result.append((c1, c2, sim_score))
			history.append({
				"c1": "c%s" % c1.cluster_id,
				"c2": "c%s" % c2.cluster_id,
				"value": round(sim_score, 3)
			})

		# Used for pretty printing cluster merging
		self.merge_history[str(self._merge_pass)] = {
			"cluster_ids": sorted(list(cluster_ids)),
			"scores": history
		}

		self._merge_pass += 1
		if len(result):
			sorted_result = sorted(result, key=lambda t: t[2], reverse=True)
			return sorted_result[0]
		else:
			return None

	def print_summary(self):
		print("*** %s ***" % (self,))
		for cluster in self.clusters:
			print(str(cluster))

	def serialize(self):
		"Extract the signatures that allows to match decks to clusters"
		return {cluster.cluster_id: cluster.serialize() for cluster in self.clusters}


class Cluster:
	"""A single cluster entity"""

	def __init__(self, cluster_set, cluster_id, decks, parents=None, parent_similarity=None):
		self._parents = parents
		self._parent_similarity = parent_similarity
		self._cluster_set = cluster_set
		self.cluster_id = cluster_id
		self._decks = decks
		self.deck_count = float(sum(1 for d in self._decks))

		cards_in_cluster = set()
		self._deck_counts_for_card = {}

		for deck in self._decks:
			for card, count in deck['cards'].items():
				cards_in_cluster.add(card)
				if card not in self._deck_counts_for_card:
					self._deck_counts_for_card[card] = float(sum(1 for d in self._decks if card in d['cards']))

		self._common_cards = {} # common across the class
		self._discarded_cards = {} # odd balls

		CORE_CUTOFF = self._cluster_set.CORE_CARD_DECK_THRESHOLD
		TECH_CUTOFF = self._cluster_set.TECH_CARD_DECK_THRESHOLD

		self.signature = {
			"core": {},
			"tech": {}
		}

		for card in cards_in_cluster:
			prevalence = self._deck_counts_for_card[card] / self.deck_count

			# card is shared among all cluster
			if card in self._cluster_set.common_cards:
				self._common_cards[card] = prevalence
				continue

			# card core to the cluster
			if prevalence >= CORE_CUTOFF:
				self.signature['core'][card] = prevalence
				continue

			# card that is likely used as a tech card
			if prevalence >= TECH_CUTOFF:
				self.signature['tech'][card] = prevalence
				continue

			# odd ball, discarding
			self._discarded_cards[card] = prevalence

	def __repr__(self):
		return str(self)

	def __str__(self):
		return self.lineage(0)

	def as_str(self):
		return "cluster %s (%i decks): %s" % (self.full_cluster_id, self.deck_count, self.pretty_signature)

	def serialize(self):
		"Return the cluster signature"
		return {
			"name": "",
			"core_cards": self.signature['core'],
			"core_cards_name": self.pretty_core_cards,
			"tech_cards": self.signature['tech'],
			"tech_cards_name": self.pretty_tech_cards,
			"observations": self.observations,
			"prevalence": self.prevalence,
			"num_decks": self.deck_count,
			"win_rate": self.win_rate
		}

	def lineage(self, depth):
		result = ""
		if depth:
			result += "\n"
		result += "\t" * depth
		result += self.as_str()
		if self._parents:
			next_depth = depth + 1
			for parent in self._parents:
				result += parent.lineage(next_depth)

		return result

	@property
	def observations(self):
		return sum(d['observations'] for d in self._decks)

	@property
	def win_rate(self):
		wr = []
		for d in self._decks:
			wr.append(d['win_rate'])
		wr = np.array(wr)
		return {
			"mean": np.mean(wr, axis=0),
			"stddev": np.std(wr, axis=0),
			"max": np.max(wr, axis=0),
			"min": np.min(wr, axis=0)
		}

	@property
	def full_cluster_id(self):
		if self._parent_similarity:
			return "{id} (Sim: {sim:>5})".format(
				id=self.cluster_id,
				sim=round(self._parent_similarity, 3)
			)
		else:
			return self.cluster_id

	@property
	def prevalence(self):
		cutoff = self._cluster_set.low_observation_volume_cutoff
		if self.observations > cutoff:
			return "common"
		else:
			return "rare"

	@property
	def core_cards(self):
		return [self._cluster_set.card_name(c) for c in self.signature["core"].keys()]

	@property
	def pretty_core_cards(self):
		return {self._cluster_set.card_name(c): round(p, 2) for c,p in self.signature["core"].items()}

	@property
	def tech_cards(self):
		return [self._cluster_set.card_name(c) for c in self.signature["tech"].keys()]

	@property
	def pretty_tech_cards(self):
		return {self._cluster_set.card_name(c): round(p, 2) for c,p in self.signature["tech"].items()}

	@property
	def discarded_cards(self):
		return [self._cluster_set.card_name(c) for c in self._discarded_cards.keys()]

	@property
	def pretty_signature(self):
		result = {self._cluster_set.card_name(c):round(p, 3) for c, p in self.signature["core"].items()}
		result.update({self._cluster_set.card_name(c):round(p, 3) for c, p in self.signature["tech"].items()})
		return result

	def signature_match(self, card_list):
		# Given a new card_list the similarity score is calculated based on the cluster signature
		raise NotImplementedError("Implement Me!")


def get_clusters(input_data):
	clusters = {}
	for player_class in CardClass:
		if CardClass.DRUID <= player_class <= CardClass.WARRIOR:
			input_map = {int(k): v for k, v in input_data["map"].items()}
			cluster_set = ClusterSet(
				player_class,
				input_map,
				input_data["decks"].get(str(int(player_class)),[])
			)
			clusters[player_class.name] = cluster_set

	return clusters
