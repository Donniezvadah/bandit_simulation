# bandits/bernoulli_bandit.py
import numpy as np

class BernoulliBandit:
    """
    A Bernoulli bandit with K actions. Each action yields a reward of 1 with probability theta_k
    and 0 otherwise, where theta_k is unknown to the agent but fixed over time.
    """

    def __init__(self, n_actions=10):
        """
        Initializes the Bernoulli bandit.

        Args:
            n_actions (int): The number of available actions (arms).
        """
        self._probs = np.random.random(n_actions)  # Random probabilities for each action
        self._initial_probs = np.copy(self._probs)  # Store initial probabilities for resetting
        
    @property
    def action_count(self):
        """Returns the number of actions."""
        return len(self._probs)

    def pull(self, action):
        """
        Simulates pulling a lever (taking an action) and returns a reward.

        Args:
            action (int): The index of the action to take.

        Returns:
            float: 1.0 if a random number is less than the action's probability, 0.0 otherwise.
        """
        if not (0 <= action < self.action_count):
            raise ValueError(f"Action {action} is out of bounds.  Must be between 0 and {self.action_count - 1}")

        if np.random.random() > self._probs[action]:
            return 0.0
        return 1.0

    def optimal_reward(self):
        """
        Returns the expected reward of the optimal action. Used for regret calculation.

        Returns:
            float: The maximum probability among all actions.
        """
        return np.max(self._probs)

    def step(self):
        """
        Used in non-stationary versions of the bandit to change the probabilities.
        This implementation is stationary, so this method does nothing.
        """
        pass  # Stationary bandit, so no need to change probabilities

    def reset(self):
        """Resets the bandit to its initial state (initial probabilities)."""
        self._probs = np.copy(self._initial_probs)