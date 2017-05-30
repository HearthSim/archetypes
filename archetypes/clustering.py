import pprint
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
	c1_sig = c1.signature
	c1_card_list = c1_sig.keys()
	c2_sig = c2.signature
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
		self.COMMON_CARD_CLUSTER_THRESHOLD = 0.8
		# weighting for the signature coeff
		self.CORE_CARD_WEIGHT = 10.0
		# weighting for the signature coeff
		self.TECH_CARD_WEIGHT = 2.0
		self.MERGE_THRESHOLD = 0.5

		self._clusters = []
		self._all_clusters =[]

		self._common_cards_for_class = None
		self._common_card_scores = None
		self._player_class = player_class
		self._card_map = card_map
		self._decks = decks
		self._deck_matrix = []
		self.merge_history = {}
		self._merge_pass = 1

		for deck in self._decks:
			# deck is a dict with 'deck_list', 'cards', 'observations'
			vector = [0] * self.collectible_card_count
			for card, count in deck['cards'].items():
				vector[int(card)] = count
			self._deck_matrix.append(vector)

		self.make_clusters()

	def __repr__(self):
		return str(self)

	def __str__(self):
		args = (self.cluster_class_name, self.total_deck_count, self.deck_count, self.cluster_count)
		return "%s: %i deck observations of %i distinct lists across %i clusters" % args

	@property
	def clusters(self):
		return self._clusters

	@property
	def collectible_card_count(self):
		return len(self._card_map)

	@property
	def deck_count(self):
		return len(self._decks)

	@property
	def total_deck_count(self):
		return float(sum(d['observations'] for d in self._decks))

	@property
	def cluster_class_name(self):
		return self._player_class.name

	@property
	def common_cards_map(self):
		self._common_cards_for_class = {}
		for card in self._card_map.keys():
			total_decks_with_card = self.total_deck_count_for_card(card)
			total_decks = self.total_deck_count
			card_prevalance = total_decks_with_card / total_decks

			if card_prevalance >= self.COMMON_CARD_CLUSTER_THRESHOLD:
				self._common_cards_for_class[card] = card_prevalance
		return self._common_cards_for_class

	@property
	def common_card_scores(self):
		self._common_card_scores = {}
		for card in self._card_map.keys():
			total_decks_with_card = self.total_deck_count_for_card(card)
			total_decks = self.total_deck_count
			card_prevalance = total_decks_with_card / total_decks

			if card_prevalance >= self.COMMON_CARD_CLUSTER_THRESHOLD:
				self._common_card_scores[self.card_name(card)] = (total_decks_with_card, total_decks)
		return self._common_card_scores

	@property
	def heatmap_data(self):
		xValues = []
		yValues = []
		zValues = []

		num_clusters = len(self._clusters)
		cards = {}

		cluster_id = 0
		for cluster in self._clusters:
			for card, score in cluster.signature.items():
				if card not in cards:
					cards[card] = [0] * num_clusters

				if card in cluster.core_cards_map:
					cards[card][cluster_id] = 1
				else:
					cards[card][cluster_id] = .5

			cluster_id  += 1
			xValues.append(cluster_id)

		for card_id, clusters in cards.items():
			card_name = self.card_name(card_id)
			yValues.append(card_name)
			zValues.append(clusters)

		data = {
			"class": self.cluster_class_name,
			"num_clusters": len(self._clusters),
			"x": xValues,
			"y": yValues,
			"z": zValues,
			"clusters": []
		}

		for cluster in self._clusters:
			data["clusters"].append({
				"core_cards": cluster.pretty_core_cards,
				"tech_cards": cluster.pretty_tech_cards,
			})

		return data

	def print_most_common_cards(self, limit=20):
		common_cards = list(
			sorted(self.common_card_scores.items(), key=lambda t: t[1], reverse=True)
		)[:limit]
		for card, score in common_cards:
			num, denom = score
			pct = round(num / denom, 4)
			print("{card:>30} -> ({pct:<6}) {score}".format(card = card, score=score, pct=pct))

	@property
	def common_cards(self):
		return [self.card_name(c) for c in self.common_cards_map.keys()]

	@property
	def cluster_count(self):
		return float(len(self._clusters))

	def card_name(self, card_index):
		card_id = self._card_map[int(card_index)]
		return db[card_id].name.replace(",", "")

	def make_clusters(self):
		# Reset any cached intermediates here.
		self._common_cards_for_class = None
		self._clusters = []

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
			self._clusters.append(c)
			self._all_clusters.append(c)

	def total_deck_count_for_card(self, card):
		return float(sum(c.deck_count_for_card(str(card)) for c in self._clusters if str(card) in c.cards_in_cluster))

	def merge_clusters(self, distance_function=cluster_similarity):
		next_cluster_id = len(self._clusters)
		self._clusters = self._do_merge_clusters(self._clusters, distance_function, next_cluster_id)

	def _do_merge_clusters(self, clusters, distance_function, first_new_cluster_id):
		done = False
		current_clusters = list(clusters)
		next_cluster_id = first_new_cluster_id
		while not done:
			most_similar = self._most_similar_pair(current_clusters, distance_function)
			if most_similar and most_similar[2] >= self.MERGE_THRESHOLD:
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
				self._all_clusters.append(new_cluster)
				next_cluster_id += 1
				next_clusters_list = [new_cluster]
				for c in current_clusters:
					if c.cluster_id not in (c1.cluster_id, c2.cluster_id):
						next_clusters_list.append(c)
				current_clusters = next_clusters_list
			else:
				done = True
		return current_clusters

	def _most_similar_pair(self, clusters, distance_function):
		result = []
		history = []
		cluster_ids = set()
		for c1, c2 in combinations(clusters, 2):
			cluster_ids.add("c%s" % c1.cluster_id)
			cluster_ids.add("c%s" % c2.cluster_id)

			sim_score = distance_function(c1, c2)
			result.append((c1, c2, sim_score))
			history.append({
				"c1": "c%s" % c1.cluster_id,
				"c2": "c%s" % c2.cluster_id,
				"value": round(sim_score, 3)
			})

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
		for cluster in self._clusters:
			print(str(cluster))


class Cluster:
	"""A single cluster within a set of clusters"""

	def __init__(self, cluster_set, cluster_id, decks, parents=None, parent_similarity=None):
		self._parents = parents
		self._parent_similarity = parent_similarity
		self._cluster_set = cluster_set
		self.cluster_id = cluster_id
		self._decks = decks
		self._deck_count = None
		self._deck_counts_for_card = {}
		self._cards_in_cluster = None
		self._signature_dict = None

		self._core_cards = None
		self._common_core_cards = None

		self._tech_cards = None
		self._common_tech_cards = None

		self._discard_cards = None
		self._common_discarded_cards = None

	def __repr__(self):
		return str(self)

	def __str__(self):
		return self.lineage(0)

	def as_str(self):
		return "cluster %s (%i decks): %s" % (self.full_cluster_id, self.deck_count, self.pretty_signature)

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
	def full_cluster_id(self):
		if self._parent_similarity:
			return "{id} (Sim: {sim:>5})".format(
				id=self.cluster_id,
				sim=round(self._parent_similarity, 3)
			)
		else:
			return self.cluster_id

	@property
	def deck_count(self):
		if not self._deck_count:
			self._deck_count = float(sum(d['observations'] for d in self._decks))
		return self._deck_count

	@property
	def cards_in_cluster(self):
		if not self._cards_in_cluster:
			self._cards_in_cluster = set()
			for deck in self._decks:
				for card, count in deck['cards'].items():
					self._cards_in_cluster.add(card)
		return self._cards_in_cluster

	def deck_count_for_card(self, card):
		if card not in self._deck_counts_for_card:
			self._deck_counts_for_card[card] = float(sum(d['observations'] for d in self._decks if card in d['cards']))
		return self._deck_counts_for_card[card]

	@property
	def core_cards_map(self):
		if not self._core_cards:
			self._tag_cards_by_type()
		return self._core_cards

	@property
	def core_cards(self):
		return [self._cluster_set.card_name(c) for c in self.core_cards_map.keys()]

	@property
	def pretty_core_cards(self):
		return {self._cluster_set.card_name(c):p for c,p in self.core_cards_map.items()}

	@property
	def tech_cards_map(self):
		if not self._tech_cards:
			self._tag_cards_by_type()
		return self._tech_cards

	@property
	def tech_cards(self):
		return [self._cluster_set.card_name(c) for c in self.tech_cards_map.keys()]

	@property
	def pretty_tech_cards(self):
		return {self._cluster_set.card_name(c):p for c,p in self.tech_cards_map.items()}

	@property
	def discarded_card_ids(self):
		if not self._discard_cards:
			self._tag_cards_by_type()
		return self._discard_cards

	@property
	def discarded_cards(self):
		return [self._cluster_set.card_name(c) for c in self._discarded_cards.keys()]

	@property
	def signature(self):
		if not self._signature_dict:
			self._signature_dict = {}
			for card, prevalance in self.core_cards_map.items():
				self._signature_dict[card] = prevalance * self._cluster_set.CORE_CARD_WEIGHT

			for card, prevalance in self.tech_cards_map.items():
				self._signature_dict[card] = prevalance * self._cluster_set.TECH_CARD_WEIGHT

		return self._signature_dict

	@property
	def pretty_signature(self):
		return {self._cluster_set.card_name(c):round(p, 3) for c, p in self.signature.items()}

	@property
	def card_heatmap_data(self):
		result = []
		card_names = []
		for card_index, card_id in self._cluster_set._card_map.items():

			if str(card_index) in self.signature:
				card_names.append(self._cluster_set.card_name(card_index))
				result.append({
					"cluster_id": "c%s" % self.cluster_id,
					"card_name": self._cluster_set.card_name(card_index),
					"value": self.signature[str(card_index)]
				})

		return card_names, result

	def _tag_cards_by_type(self):
		self._core_cards = {}
		self._common_core_cards = []

		self._tech_cards = {}
		self._common_tech_cards = []

		self._discarded_cards = {}
		self._common_discarded_cards = []

		CORE_CUTOFF = self._cluster_set.CORE_CARD_DECK_THRESHOLD
		TECH_CUTOFF = self._cluster_set.TECH_CARD_DECK_THRESHOLD
		common_cards = self._cluster_set.common_cards_map.keys()

		for card in self.cards_in_cluster:
			prevalence = self.deck_count_for_card(card) / self.deck_count
			if prevalence >= CORE_CUTOFF:
				if card not in common_cards:
					self._core_cards[card] = prevalence
				else:
					self._common_core_cards.append(card)
			elif prevalence >= TECH_CUTOFF:
				if card not in common_cards:
					self._tech_cards[card] = prevalence
				else:
					self._common_tech_cards.append(card)
			else:
				if card not in common_cards:
					self._discarded_cards[card] = prevalence
				else:
					self._common_discarded_cards.append(card)

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
