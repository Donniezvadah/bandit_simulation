# bandits/main.py
import numpy as np
from collections import OrderedDict
from bernoulli_bandit import BernoulliBandit
from epsilon import EpsilonGreedyAgent
from ucb import UCBAgent
from thompson import ThompsonSamplingAgent
from plots import plot_regret
import matplotlib.pyplot as plt

def get_regret(env, agents, n_steps=10_000, n_trials=100):
    """
    Simulates the multi-armed bandit problem for a given environment and a set of agents.

    Args:
        env (BernoulliBandit): The bandit environment.
        agents (list): A list of AbstractAgent instances to evaluate.
        n_steps (int): The number of steps (time horizon) for each trial.
        n_trials (int): The number of independent trials to run.

    Returns:
        OrderedDict: An OrderedDict where keys are agent names and values are lists of cumulative regret.
    """
    scores = OrderedDict({
        agent.name: [0.0 for _ in range(n_steps)] for agent in agents
    })

    for trial in range(n_trials):
        env.reset()

        for agent in agents:
            agent.init_actions(env.action_count)

        for i in range(n_steps):
            optimal_reward = env.optimal_reward()

            for agent in agents:
                action = agent.get_action()
                reward = env.pull(action)
                agent.update(action, reward)
                scores[agent.name][i] += optimal_reward - reward

            env.step()  # Change bandit's state if it is unstationary

    for agent in agents:
        scores[agent.name] = np.cumsum(scores[agent.name]) / n_trials

    return scores


def main():
    """
    Main function to run the bandit simulation and plot the results.
    """
    np.set_printoptions(precision=3, suppress=True) # set print format

    # Initialize the bandit environment
    bandit = BernoulliBandit()

    # Initialize the agents
    agents = [
        EpsilonGreedyAgent(),
        UCBAgent(),
        ThompsonSamplingAgent()
    ]

    # Run the simulation and get the regret scores
    regret = get_regret(bandit, agents, n_steps=10000, n_trials=10)

    # Plot the regret
    plot_regret(agents, regret)


if __name__ == "__main__":
    main()