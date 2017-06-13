import numpy as np

class PrettyClusterMixin(object):

	@property
	def win_rate(self):
		wr = np.array([d["win_rate"] for d in self._decks])
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

	def __repr__(self):
		return str(self)

	def __str__(self):
		if self._has_metadata:
			return self.lineage(0)
		return "Cluster %s %s" % (self.cluster_id, self.name)

	def as_str(self):
		return "cluster %s (%i decks): %s" % (
			self.full_cluster_id, self._num_decks, self.pretty_signature
		)

	@property
	def core_cards(self):
		return [self._player_class_cluster.card_name(c) for c in self.cards["core"].keys()]

	@property
	def pretty_core_cards(self):
		return {self._player_class_cluster.card_name(c): round(p, 2) for c,p in self.cards["core"].items()}

	@property
	def tech_cards(self):
		return [self._player_class_cluster.card_name(c) for c in self.cards["tech"].keys()]

	@property
	def pretty_tech_cards(self):
		return {self._player_class_cluster.card_name(c): round(p, 2) for c,p in self.cards["tech"].items()}

	@property
	def discarded_cards(self):
		return [self._player_class_cluster.card_name(c) for c in self.cards["discarded"].items()]

	@property
	def pretty_signature(self):
		result = {self._player_class_cluster.card_name(c):round(p, 3) for c, p in self.cards["core"].items()}
		result.update({self._player_class_cluster.card_name(c):round(p, 3) for c, p in self.cards["tech"].items()})
		return result



class PrettyPlayerClassClustersMixin(object):

	@property
	def heatmap_data(self):
		cluster_ids = []
		zValues = []

		card_list = []
		num_clusters = len(self.clusters)
		cards = {}

		cluster_id = 0
		for cluster in self.clusters:
			for ctype, members in cluster.cards.items():
				for card, value in members.items():
					if card not in cards:
						cards[card] = [0] * num_clusters
					if ctype == "core":
						cards[card][cluster_id] = 1
					elif ctype == "tech":
						cards[card][cluster_id] = .5

			cluster_id  += 1
			cluster_ids.append(cluster_id)

		for card_id, clusters in cards.items():
			card_name = self.card_name(card_id)
			card_list.append(card_name)
			zValues.append(clusters)

		common_cards = [self.card_name(c) for c in self.common_cards.keys()]
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
				"num_decks": cluster._num_decks,
				"num_observations": cluster.observations,
				"prevalence": cluster.prevalence,
				"winrate": cluster.win_rate
			})

		clusters_for_data = sorted(clusters_for_data, key=lambda c: c["num_observations"], reverse=True)
		data["clusters"] = clusters_for_data
		return data

	def get_by_id(self, cluster_id):
		for cluster in self.clusters:
			if cluster.cluster_id == cluster_id:
				return cluster
		return None

	@property
	def cluster_class_name(self):
		return self.player_class.name

	@property
	def cluster_count(self):
		return float(len(self.clusters))


	def __repr__(self):
		return str(self)

	def __str__(self):
		args = (self.cluster_class_name, self.num_decks, self.cluster_count)
		return "%s: %i deck observations across %i clusters" % args

	def print_most_common_cards(self):
		for card, score in self.common_cards:
			print("{card:>30} -> {score}".format(card = card, score=score))

	def print_summary(self):
		print("*** %s ***" % (self,))
		for cluster in self.clusters:
			print(str(cluster))
