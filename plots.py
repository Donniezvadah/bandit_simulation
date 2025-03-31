# bandits/plots.py
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict


def plot_regret(agents, scores):
    """
    Plots the cumulative regret of each agent over time.

    Args:
        agents (list): A list of AbstractAgent instances.
        scores (OrderedDict): An OrderedDict where keys are agent names and values are lists of cumulative regret.
    """
    for agent in agents:
        plt.plot(scores[agent.name], linewidth=1)

    plt.legend([agent.name for agent in agents])
    plt.ylabel("Expected Cumulative Regret")
    plt.xlabel(r"$T$")
    plt.title("Expected Cumulative Regret vs. Steps")  # Added a title
    plt.grid(False) #Added gridlines
    plt.show()
    
    plt.savefig("regret.png")
    plt.savefig("regret.pdf")
    
    
    