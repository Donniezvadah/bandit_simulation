# bandits/ucb.py
import numpy as np
import math
from abstract_agent import AbstractAgent
# from abc import ABCMeta, abstractmethod, abstractproperty

class UCBAgent(AbstractAgent):
    """
    An agent that uses the Upper Confidence Bound (UCB1) algorithm to select actions.
    UCB1 balances exploration and exploitation by adding an exploration bonus to the estimated
    reward of each action.
    """

    def __init__(self):
        self._successes = None  # Initialize successes and failures to None
        self._failures = None

    def init_actions(self, n_actions):
        """
        Initializes the agent's internal state (successes and failures counts for each action).

        Args:
            n_actions (int): The number of possible actions.
        """
        super().init_actions(n_actions) # Calls parent initialization function
        self._successes = np.zeros(n_actions)
        self._failures  = np.zeros(n_actions)

    def get_action(self):
        """
        Chooses an action based on the UCB1 algorithm.

        Returns:
            int: The index of the chosen action.
        """
        if self._successes is None or self._failures is None:
            raise ValueError("Agent has not been initialized. Call init_actions() first.")

        n_actions = self._successes + self._failures
        #Handle cases of actions never taken.
        if np.any(n_actions == 0):
            return np.argmin(n_actions) # Select any action that hasn't been taken.

        ucb = np.sqrt(2 * np.log(self._total_pulls) / n_actions) # Use log, not log10
        p = self._successes / n_actions + ucb
        return np.argmax(p)

    @property
    def name(self):
        """Returns the name of the agent."""
        return self.__class__.__name__