from typing import cast
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sjlearn.planets_simulator import PlanetsBehavior, estimate_phase

pb = PlanetsBehavior()
pb.reset()

fig, axes = cast(tuple[Figure, tuple[Axes, Axes, Axes]], plt.subplots(3))
X = np.arange(100)

# sample points with simulation
nb_samples = 25
chosen_planets = np.random.randint(low=0, high=3, size=(nb_samples,),
                                   dtype=np.int8)
mortys_have_survived: list[list[bool]] = [[], [], []]
planet_periods = list(2 * np.pi / np.array(pb.periodOnPlanets))
for step in range(nb_samples):
    chosen_planet = chosen_planets[step]
    has_one_morty_survived = pb.simulatePlanet(pb.function_of_planet(
        chosen_planet, step
    ))
    mortys_have_survived[chosen_planet].append(has_one_morty_survived)

estimated_phase = estimate_phase(mortys_have_survived, planet_periods,
                                 chosen_planets)
print("Estimated phase:", estimated_phase.item())
print("Actual phase:", pb.initialPhaseOnPlanets)

for planet, ax in enumerate(axes):
    fn = pb.function_of_planet(planet)
    estimate_fn = pb.function_of_planet(
        planet,
        try_phase=estimated_phase.item()
    )
    ax.plot(X, fn[:100], label="true")
    ax.plot(X, estimate_fn[:100], label="estimated")
    ax.legend()
    ax.set_title(f"Planet {planet + 1} (period {pb.periodOnPlanets[planet]})")

fig.savefig(Path("plots") / "my_curves.png")
