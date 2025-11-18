"""Simulate the behavior of the planets.
"""

from typing import cast
from .types import MortysSentOnPlanet, Planet
import numpy as np

MortysOnPlanetState = np.ndarray[tuple[int], np.dtype[np.int_]]

class PlanetsBehavior:
    AVERAGE_SURVIVAL_RATE = 0.5  # constant according to Rick
    MAX_MORTYS_NB = 1000
    def __init__(self) -> None:
        self.perPlanetMortysSent: MortysOnPlanetState = np.array(
            [0, 0, 0], dtype=np.int_
        )
        self.episodeErrorSeed = 42
        self.totalSentMorties = 0
        # the model below is set from a data visualization
        self.periodOnPlanets = [10, 20, 200]  # in number of mortys
        self.amplitudeOnPlanets = [0.5, 0.5, 0.5]

        # these are the unknown values to be discovered during one episode
        # in training we simulate them
        self.initialPhaseOnPlanets = [0, 0, 0]


    def reset(self):
        self.perPlanetMortysSent = np.array([0, 0, 0], dtype=np.int_)
        self.initialPhaseOnPlanets = [0, 0, 0]
        self.totalSentMorties = 0
        return self.perPlanetMortysSent, None

    def survivalRate(self, planet: Planet) -> float:
        # according to the indices, the survival rate oscillate around 0.5
        # according to the number of mortys sent on the planet
        # the period depends on the planet and has been observed with data
        # visualization
        t = self.perPlanetMortysSent[planet]
        return (
            PlanetsBehavior.AVERAGE_SURVIVAL_RATE +
                self.amplitudeOnPlanets[planet] * np.cos(
                    self.periodOnPlanets[planet]*t +
                        self.initialPhaseOnPlanets[planet]
                )
        )

    def simulatePlanet(self, survivalRate: float) -> bool:
        # sample if the Mortys survive or not
        # the probe is around the survival rate (with a small normal error)
        SMALL_NORMAL_ERROR = 0.01
        def restrict(val: float, min_: float, max_: float) -> float:
            return max(min_, min(max_, val))
        small_error = restrict(SMALL_NORMAL_ERROR * np.random.randn(),
                               -SMALL_NORMAL_ERROR, SMALL_NORMAL_ERROR)
        return np.random.rand() <= restrict(survivalRate + small_error, 0, 1)

    def step(self, action: int):
        planet, mortysSent = cast(
            tuple[Planet, MortysSentOnPlanet],
            (action // 3, 1 + action % 3)
        )
        assert (mortysSent <= PlanetsBehavior.MAX_MORTYS_NB - self.totalSentMorties), "Too much mortys"
        reward = mortysSent * self.simulatePlanet(self.survivalRate(planet))
        self.perPlanetMortysSent[planet] += mortysSent
        self.totalSentMorties += mortysSent
        next_state, reward, terminated, truncated, nothing = (
            self.perPlanetMortysSent,
            reward,
            self.totalSentMorties == PlanetsBehavior.MAX_MORTYS_NB,
            False,
            None
        )
        return next_state, reward, terminated, truncated, nothing

