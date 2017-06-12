import json
import pprint
import numpy as np
from itertools import combinations
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hearthstone.enums import CardClass
from hearthstone.cardxml import load
from .mixins import PrettyClusterMixin, PrettyPlayerClassClustersMixin


pp = pprint.PrettyPrinter(indent=4)
db, _ = load()

def sanitize_name(name):
	return name.replace(u"\u2019", u"'").encode("ascii", "ignore")

def cluster_similarity(c1, c2):
	"Compute a weighted similarity based of the signatures"

	## Maybe use tech cards as well?
	c1_sig = c1.cards['core']
	c1_card_list = c1_sig.keys()
	c2_sig = c2.cards['core']
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


class ClusterSet(object):

	def __init__(self, player_class_clusters):
		self.player_class_clusters = player_class_clusters

	@classmethod
	def from_input_data(cls, input_data):
		clusters = {}
		for player_class in CardClass:
			if CardClass.DRUID <= player_class <= CardClass.WARRIOR:
				input_map = {int(k): v for k, v in input_data["map"].items()}
				cluster_set = PlayerClassClusters.create(
					player_class,
					input_map,
					input_data["decks"].get(str(int(player_class)), [])
				)
				clusters[player_class.name] = cluster_set

		return ClusterSet(clusters)

	@classmethod
	def from_file(cls, fp):
		data = json.loads(fp.read())
		player_class_clusters = {}
		for player_class, serialized_clusters_obj in data.items():
			player_class_clusters[player_class] = PlayerClassClusters.deserialize(serialized_clusters_obj)
		return ClusterSet(player_class_clusters)

	def save(self, fp):
		for_json = {k:v.serialize() for k,v in self.player_class_clusters.items()}
		fp.write(json.dumps(for_json, indent=4))

	def update(self, new_cluster_set):
		new_player_class_clusters = new_cluster_set.player_class_clusters.copy()

		for player_class, cluster_set in new_player_class_clusters.items():
			prev_cluster_set = self.player_class_clusters[player_class]
			for cluster in cluster_set.clusters:
				prev_cluster = prev_cluster_set.find_cluster(cluster)
				if prev_cluster is None:
					print("New %s cluster!" % player_class)
				else:
					cluster.name = prev_cluster.name

		#TODO: report dropped clusters

		return ClusterSet(new_player_class_clusters)


NUM_CLUSTERS_REQUESTED = 10
# % of deck within a cluster that need to have the card to be considered core card
CORE_CARD_DECK_THRESHOLD = 0.8
# % of deck within a cluster that need to have a card to be considerd a tech card
TECH_CARD_DECK_THRESHOLD = 0.3
# % of decks ehat need to contains the card as to count it as common_card accross the class (remove it from signatures)
COMMON_CARD_CLUSTER_THRESHOLD = 0.93
LOW_VOLUME_CLUSTER_MULTIPLIER = 1.5


class PlayerClassClusters(PrettyPlayerClassClustersMixin):
	"""A container for a class's deck clusters."""

	def __init__(self, clusters=None):
		self.clusters = clusters if clusters else []
		self._merge_history = {}
		self._merge_pass = 1

	@classmethod
	def create(cls, player_class, card_map, decks):
		instance = PlayerClassClusters()
		instance._card_map = card_map
		instance.player_class = player_class

		common_cards = defaultdict(int)
		for deck in decks:
			for card, count in deck["cards"].items():
				common_cards[card] += 1

		num_decks = float(len(decks))
		for card, count in list(common_cards.items()):
			prevalence = float(count) / num_decks
			if prevalence < COMMON_CARD_CLUSTER_THRESHOLD:
				del common_cards[card]

		instance.common_cards = common_cards
		instance.num_decks = num_decks

		deck_matrix = []
		for deck in decks:
			vector = [0] * len(card_map)
			for card, count in deck["cards"].items():
				if card not in common_cards:
					vector[int(card)] = count
			deck_matrix.append(vector)

		# print("deck matrix", deck_matrix)
		# Then do clustering work
		x = StandardScaler().fit_transform(deck_matrix)

		# TODO find good random state, other way go get consistent result
		# or a way do deal with inconsitent results
		clusterizer = KMeans(
			n_clusters=min(NUM_CLUSTERS_REQUESTED, len(deck_matrix)),
			random_state=827466192
		)
		clusterizer.fit(x)

		clustered_decks = defaultdict(list)
		for deck, cluster_id in zip(decks, clusterizer.labels_):
			clustered_decks[cluster_id].append(deck)

		for cluster_id, decks in clustered_decks.items():
			cluster = Cluster.create(instance, str(cluster_id), decks, common_cards)
			instance.clusters.append(cluster)

		na = np.array([cluster.observations for cluster in instance.clusters])
		avg = np.mean(na, axis=0)
		observation_cutoff = avg / LOW_VOLUME_CLUSTER_MULTIPLIER

		instance._merge_clusters()

		for cluster in instance.clusters:
			cluster.prevalence = "rare" if cluster.observations < observation_cutoff else "common"

		return instance

	def card_name(self, card_index):
		card_id = self._card_map[int(card_index)]
		return sanitize_name(db[card_id].name.replace(",", ""))

	@classmethod
	def deserialize(cls, json_obj):
		clusters = [Cluster.deserialize(cluster_data) for cluster_data in json_obj]
		return PlayerClassClusters(clusters)

	def serialize(self, legacy=False):
		"Extract the signatures that allows to match decks to clusters"
		if legacy:
			return {cluster.cluster_id: cluster.serialize() for cluster in self.clusters}
		return [cluster.serialize() for cluster in self.clusters]

	def _merge_clusters(self, distance_function=cluster_similarity):
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
					pass
					# print("%s - %s" % (c1.cluster_id, distances))
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
		observation_threshold = mean / LOW_VOLUME_CLUSTER_MULTIPLIER
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
			new_cluster = Cluster.create(
				self,
				next_cluster_id,
				new_cluster_decks,
				self.common_cards,
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
		self._merge_history[str(self._merge_pass)] = {
			"cluster_ids": sorted(list(cluster_ids)),
			"scores": history
		}

		self._merge_pass += 1
		if len(result):
			sorted_result = sorted(result, key=lambda t: t[2], reverse=True)
			return sorted_result[0]
		else:
			return None

	def find_cluster(self, target_cluster):
		for cluster in self.clusters:
			if cluster.equals(target_cluster):
				return cluster
		return None



class Cluster(PrettyClusterMixin):
	"""A single cluster entity"""

	def __init__(self, cluster_id, name, cards, observations, prevalence):
		self.cluster_id = cluster_id
		self.name = name
		self.cards = cards
		self.observations = observations
		self.prevalence = prevalence
		self._has_metadata = False

	@classmethod
	def create(cls, player_class_cluster, cluster_id, decks, common_cards, parents=None, parent_similarity=None):

		# extract data from decks
		observations = 0
		deck_counts_for_card = {}
		all_cards = set()
		for deck in decks:
			observations += deck["observations"]
			for card, _ in deck["cards"].items():
				all_cards.add(card)
				if card not in deck_counts_for_card:
					deck_counts_for_card[card] = sum(1 for d in decks if card in d["cards"])

		# group cards
		deck_count = len(decks)
		cards = {"core": {}, "tech": {}, "common": {}, "discarded": {}}
		for card in all_cards:
			prevalence = float(deck_counts_for_card[card]) / deck_count
			if card in common_cards:
				cards["common"][card] = prevalence
			elif prevalence >= CORE_CARD_DECK_THRESHOLD:
				cards["core"][card] = prevalence
			elif prevalence >= TECH_CARD_DECK_THRESHOLD:
				cards["tech"][card] = prevalence
			else:
				cards["discarded"][card] = prevalence

		# create cluster
		cluster = Cluster(
			cluster_id, None, cards, observations, prevalence
		)

		# add debug fields
		cluster._has_metadata = True
		cluster._player_class_cluster = player_class_cluster
		cluster._parents = parents
		cluster._parent_similarity = parent_similarity
		cluster._decks = decks
		cluster._num_decks = deck_count

		return cluster

	def serialize(self):
		serialized = {
			"name": self.name,
			"cluster_id": self.cluster_id,
			"core_cards": self.cards["core"],
			"tech_cards": self.cards["tech"],
			"observations": self.observations,
			"prevalence": self.prevalence,
		}
		if self._has_metadata:
			serialized.update({
				"win_rate": self.win_rate,
				"core_card_names": self.pretty_core_cards,
				"tech_card_names": self.pretty_tech_cards
			})
		return serialized

	@classmethod
	def deserialize(cls, data):
		return Cluster(
			data["cluster_id"],
			data["name"],
			{"core": data["core_cards"], "tech": data["tech_cards"]},
			data["observations"],
			data["prevalence"]
		)

	def signature_match(self, card_list):
		# Given a new card_list the similarity score is calculated based on the cluster signature
		raise NotImplementedError("Implement Me!")

	def equals(self, cluster):
		for card_set in ["core", "tech"]:
			if len(self.cards[card_set].keys()) != len(cluster.cards[card_set].keys()):
				return False
			for card_id, prevalence in self.cards[card_set].items():
				if card_id not in cluster.cards[card_set]:
					return False
				if cluster.cards[card_set][card_id] != prevalence:
					return False
		return True

