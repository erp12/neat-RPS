import numpy as np
import random

from rock_paper_stuff import RESOURCE_NAMES
from rock_paper_stuff.player import Player


class ANNPlayer(Player):

    def __init__(self, model, name):
        super().__init__(name)
        self.model = model

    def _make_feature_vec(self, other_inventory):
        player_res = []
        other_res = []
        for res in RESOURCE_NAMES:
            player_res.append(self.inventory[res])
            other_res.append(other_inventory[res])
        return np.array(player_res + other_res)

    def _get_trade_from_scores(self, class_scores):
        ndx = np.argmax(class_scores)
        return RESOURCE_NAMES[ndx]

    # def _get_inventory_deviance(self) -> float:
    #     res_counts = np.array(list(self.inventory.values()))
    #     if np.sum(res_counts) == 0:
    #         return 1e10
    #     return np.std(res_counts)

    def strategy(self, other_player):

        features = self._make_feature_vec(other_player.inventory)
        features = np.append(features, np.array([int(other_player.name == "SIMPLE")]))
        scores = self.model.activate(features)
        trade = self._get_trade_from_scores(scores)

        # If model says to trade a resource with ANNPlayer does not have
        # it will fall back on trading randomly.
        if self.inventory[trade] == 0:
            options = []
            for resource in self.inventory.keys():
                if self.inventory[resource] > 0:
                    options.append(resource)
            return random.choice(options)

        return trade
