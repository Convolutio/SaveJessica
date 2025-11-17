from .planets_simulator import PlanetsBehavior
from .dqn_prioritised_experience_replay import DQNPrioritisedExpReplayAgent, train

from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

def problem_attributes():
    state_dim = 3  # number of Mortys per planets
    action_dim = 9  # number of Mortys to be sent on a planet in one round
    return state_dim, action_dim

def checkpoint_path():
    return Path("data") / "train_checkpoint.pt"

def train_model():
    state_dim, action_dim = problem_attributes()
    planet_behavior = PlanetsBehavior()
    num_episodes = 30
    max_steps_per_episode = PlanetsBehavior.MAX_MORTYS_NB 
    target_survival_rate = 0.8  # we hope
    mortySenderAgent, scores = train(
        planet_behavior, state_dim, action_dim, num_episodes,
        max_steps_per_episode,
        int(target_survival_rate*PlanetsBehavior.MAX_MORTYS_NB)
    )
    mortySenderAgent.save(checkpoint_path())
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(scores)), scores)
    ax.set_title("Scores through the steps and the episodes.")
    fig.savefig(Path("plots") / "train_scores.png")

def infer_model():
    state_dim, action_dim = problem_attributes()
    mortySenderAgent = DQNPrioritisedExpReplayAgent(state_dim, action_dim)
    mortySenderAgent.load(checkpoint_path())

