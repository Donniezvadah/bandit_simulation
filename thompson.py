# bandits/thompson.py
import numpy as np
from abstract_agent import AbstractAgent
# from abc import ABCMeta, abstractmethod, abstractproperty

class ThompsonSamplingAgent(AbstractAgent):
    """
    An agent that uses Thompson Sampling to select actions. Thompson Sampling maintains a Beta
    distribution for each action, representing the uncertainty about its reward probability, and
    samples from these distributions to choose the next action.
    """

    def __init__(self):
        self._successes = None # Initialize successes and failures to None
        self._failures = None

    def init_actions(self, n_actions):
        """
        Initializes the agent's internal state (successes and failures counts for each action).

        Args:
            n_actions (int): The number of possible actions.
        """
        super().init_actions(n_actions) # Initialization of _successes and _failures
        self._successes = np.zeros(n_actions) # Initialization of _successes and _failures
        self._failures = np.zeros(n_actions)

    def get_action(self):
        """
        Chooses an action based on Thompson Sampling.

        Returns:
            int: The index of the chosen action.
        """

        if self._successes is None or self._failures is None:
            raise ValueError("Agent has not been initialized. Call init_actions() first.")

        #theta = np.array([np.random.beta(se
        # lf._successes[i], self._failures[i]) if self._successes[i]!=0 and self._failures[i]!=0 else np.random.random() for i in range(len(self._successes))])
        #Replace random.random() to avoid always selecting the first action at the beginning
        theta = np.array([np.random.beta(self._successes[i] + 1e-6, self._failures[i] + 1e-6) for i in range(len(self._successes))])

        return np.argmax(theta)

    @property
    def name(self):
        """Returns the name of the agent."""
        return self.__class__.__name__