"""Simulate the behavior of the planets.
"""

from typing import Optional, cast, overload
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
        self.amplitudeOnPlanets = 0.5  # same on all planet

        # these are the unknown values to be discovered during one episode
        # in training we simulate them
        self.initialPhaseOnPlanets = .0
        self.initialPhaseCoeffs = [1., .0]


    def reset(self):
        self.perPlanetMortysSent = np.array([0, 0, 0], dtype=np.int_)
        self.initialPhaseOnPlanets = np.pi*(-1 + 2*np.random.rand())
        self.initialPhaseCoeffs: list[float] = [
            np.cos(self.initialPhaseOnPlanets),
            -np.sin(self.initialPhaseOnPlanets)
        ]
        self.totalSentMorties = 0
        return self.perPlanetMortysSent, None

    @overload
    def function_of_planet(self, planet: Planet, t: None=None, try_phase: Optional[float]=None) -> np.ndarray[tuple[int],
        np.dtype[np.floating]]:
        pass

    @overload
    def function_of_planet(self, planet: Planet, t: int,
                           try_phase: Optional[float]=None) -> float:
        pass

    def function_of_planet(self, planet: Planet, t: Optional[int]=None,
                           try_phase: Optional[float]=None) -> np.ndarray[tuple[int],
        np.dtype[np.floating]] | float:
        parsed_t = t if t is not None else np.arange(1000)
        phase = try_phase if try_phase is not None else self.initialPhaseOnPlanets
        arg = (
            np.pi * (2 * parsed_t) / self.periodOnPlanets[planet] + phase
        )
        return (
            PlanetsBehavior.AVERAGE_SURVIVAL_RATE +
                self.amplitudeOnPlanets * np.cos(arg)
        )

    def survivalRate(self, planet: Planet) -> float:
        # according to the indices, the survival rate oscillate around 0.5
        # according to the number of mortys sent on the planet
        # the period depends on the planet and has been observed with data
        # visualization
        t = self.totalSentMorties
        return self.function_of_planet(planet, t)

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

import numpy as np

type MortysSents = list[bool]
type ChosenPlanets = np.ndarray[tuple[int], np.dtype[np.int8]]  # in [0; 2]

def estimate_phase(y_groups: list[MortysSents], omega_list: list[float],
                     t_groups: ChosenPlanets,
                     weights: Optional[list[int]]=None):
    """
    y_groups: list of arrays of 0/1 samples, one per channel
    omega_list: list/array of omegas (same length)
    t_groups: list of arrays of time indices/timestamps corresponding to each y array
    weights: optional list of weights for channels (default: number of samples)
    returns: a_hat in [0, 2*pi)
    """
    K = len(y_groups)
    if weights is None:
        weights = [len(y) for y in y_groups]
    Ctot = 0+0j
    for k in range(K):
        y = np.asarray(y_groups[k])
        t = np.argwhere(t_groups == k)[:, 0]
        z = y - 0.5
        Ck = np.sum(z * np.exp(-1j * omega_list[k] * t))
        Ctot += weights[k] * Ck

    a_hat = np.angle(Ctot)  # returns in [-pi, pi]
    if a_hat < 0:
        a_hat += 2*np.pi
    return a_hat


# def estimate_a_demod(y: np.ndarray[tuple[int], np.dtype[np.bool_]],
#                      omega: int):
#     # y: array of 0/1 samples indexed t=1..M
#     M = len(y)
#     t = np.arange(1, M+1)
#     z = y - 0.5
#     C = np.sum(z * np.exp(-1j * omega * t))
#     a_hat = np.angle(C)            # between -pi and +pi
#     if a_hat < 0:
#         a_hat += 2*np.pi
#     return a_hat
