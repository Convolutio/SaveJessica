"""Simulate the behavior of the planets.
"""

from typing import Optional, cast, overload
from .types import MortysSentOnPlanet, Planet
import numpy as np

MortysOnPlanetState = np.ndarray[tuple[int], np.dtype[np.int_]]

class PlanetsBehavior:
    AVERAGE_SURVIVAL_RATE = 0.5  # constant according to Rick
    MAX_MORTYS_NB = 1000
    PLANET_NUMBER = 3
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

        # these are the stacked information about the trips
        # and the estimation of the phasis
        self.estimatedPhaseOnPlanets = 1.0
        self.chosen_planets: list[int] = []
        self.good_trips_in_planets: list[list[bool]] = [[], [], []]


    def reset(self):
        self.perPlanetMortysSent = np.array([0, 0, 0], dtype=np.int_)
        def init_random_phase() -> float:
            # in [-pi, pi]
            return np.pi*(-1 + 2*np.random.rand())
        self.initialPhaseOnPlanets = init_random_phase()
        self.initialPhaseCoeffs: list[float] = [
            np.cos(self.initialPhaseOnPlanets),
            -np.sin(self.initialPhaseOnPlanets)
        ]
        self.totalSentMorties = 0


        # these are the stacked information about the phasis
        self.estimatedPhaseOnPlanets = init_random_phase()
        self.chosen_planets: list[int] = []
        self.good_trips_in_planets: list[list[bool]] = [[], [], []]
        return self.compute_information_state(), None


    def update_phasis_tracking_information(self, chosen_planet: Planet,
                                           has_one_morty_survived: bool,
                                           skip_phase_estimation: bool):
        # sample points with simulation
        self.chosen_planets.append(chosen_planet)
        self.good_trips_in_planets[chosen_planet].append(has_one_morty_survived)
        planet_omegas = list(2 * np.pi / np.array(self.periodOnPlanets))
        if not skip_phase_estimation:
            self.estimatedPhaseOnPlanets = estimate_phase(
                self.good_trips_in_planets, planet_omegas,
                np.array(self.chosen_planets, dtype=np.int8)
            ).item()

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
        t = self.perPlanetMortysSent[planet]
        return self.function_of_planet(planet, t)

    def estimatedSurivalRate(self, planet: Planet) -> float:
        # with the estimated phase
        t = self.totalSentMorties
        return self.function_of_planet(planet, t, try_phase=self.estimatedPhaseOnPlanets)

    def simulatePlanet(self, survivalRate: float) -> bool:
        # sample if the Mortys survive or not
        # the probe is around the survival rate (with a small normal error)
        SMALL_NORMAL_ERROR = 0.01
        def restrict(val: float, min_: float, max_: float) -> float:
            return max(min_, min(max_, val))
        small_error = restrict(SMALL_NORMAL_ERROR * np.random.randn(),
                               -SMALL_NORMAL_ERROR, SMALL_NORMAL_ERROR)
        return np.random.rand() <= restrict(survivalRate + small_error, 0, 1)

    def estimated_survival_rates(self):
        return np.array([
            self.estimatedSurivalRate(planet) for planet in range(PlanetsBehavior.PLANET_NUMBER)
        ])

    def select_best_estimated_planet(self) -> tuple[int, float]:
        rates = self.estimated_survival_rates()
        return np.argmax(rates).item(), np.max(rates).item()

    def compute_information_state(self):
        """The estimated survival rate weighted by the number of mortys on each
        planet bring information about the current state of the game.

        To keep a small space of input states, we modulo the number of mortys
        on received on a planet by the period of the planets
        """
        return (1 + self.perPlanetMortysSent % np.array(self.periodOnPlanets)) * self.estimated_survival_rates()

    def declare_step(self, planet: Planet, mortysSent: int, survived: bool,
                     skip_phase_estimation=False):
        """Call this function to update the planet model.
        """
        reward = mortysSent * survived
        self.update_phasis_tracking_information(planet, survived, skip_phase_estimation)
        self.perPlanetMortysSent[planet] += mortysSent
        self.totalSentMorties += mortysSent
        next_state, reward, terminated, truncated, nothing = (
            self.compute_information_state(),
            reward,
            self.totalSentMorties == PlanetsBehavior.MAX_MORTYS_NB,
            False,
            None
        )
        return next_state, reward, terminated, truncated, nothing

    def step(self, action: int):
        """Call this function during simulation to simulate a step and update the model.
        """
        planet, mortysSent = cast(
             tuple[Planet, MortysSentOnPlanet],
             (
                action // PlanetsBehavior.PLANET_NUMBER,
                1 + action % PlanetsBehavior.PLANET_NUMBER
            )
        )
        assert (mortysSent <= PlanetsBehavior.MAX_MORTYS_NB - self.totalSentMorties), f"Too much mortys (actual: {PlanetsBehavior.MAX_MORTYS_NB - self.totalSentMorties}, new: {mortysSent})"
        mortys_has_survived = self.simulatePlanet(self.survivalRate(planet))
        return self.declare_step(planet, mortysSent, mortys_has_survived)


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
