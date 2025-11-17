"""Implements a strategy using the UCB learning algorithm.

A modelisation of the survival rate behavior on each planet is applied inside
this algorithm.
"""

from typing import override
from api_client import SphinxAPIClient
from bandit import Bandit
from strategy import MortyRescueStrategy
import numpy as np

type Planet = int
type MortysSentOnPlanet = int

type SurvivalRatesOfPlanets = np.ndarray[
    tuple[Planet, MortysSentOnPlanet], np.dtype[np.floating]
]
"""The estimated future survival rates on the planet.

Matrix of shape (3, 9) (nb of planets, window of potentially-future nb of morties on a planet)

This data enables to take a decision on the number of morties to be sent.
"""

type ConfidenceInPlanetBehaviour = np.ndarray[tuple[Planet], np.dtype[np.floating]]
"""A metric to score how known the estimated behavior of a planet is.

Vector of shape (3,)
"""

class UCBWithGroundKnowledge(MortyRescueStrategy):
    def __init__(self, client: SphinxAPIClient):
        super().__init__(client)
        self.bandit = Bandit(k_arm=3, strategy="UCB", update_mode="average")

    def asympotical_survival_rate(self):
        pass

    def chooseMortyNumber(
        self,
        future_survival_rates: SurvivalRatesOfPlanets,
        current_confidences: ConfidenceInPlanetBehaviour
    ) -> MortysSentOnPlanet:
        """Deterministic number of 
        """
        raise NotImplementedError()

    @override
    def execute_strategy(self):
        print("\n=== EXECUTING UCB STRATEGY WITH HEURISTIC ABOUT THE GROUND ===")
        
