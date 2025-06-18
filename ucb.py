import numpy as np
import math


class UCB:
    """
    Upper Confidence Bound (UCB) algorithm for multi-armed bandit problems.

    UCB uses the principle of "optimism in the face of uncertainty".
    It selects arms based on: Q(a) + c * sqrt(ln(t) / N(a))
    where:
    - Q(a) is the estimated value of arm a
    - c is the exploration parameter
    - t is the total number of actions taken
    - N(a) is the number of times arm a has been selected

    This approach naturally balances exploration and exploitation by
    giving higher confidence bounds to less-explored arms.
    """

    def __init__(self, n_arms, c=2.0, initial_values=None):
        """
        Initialize the UCB bandit algorithm.

        Args:
            n_arms: Number of bandit arms
            c: Exploration parameter (higher c = more exploration)
            initial_values: Initial Q-values for each arm (default: zeros)
        """
        self.n_arms = n_arms
        self.c = c

        # Initialize action-value estimates (Q-values)
        if initial_values is None:
            self.q_values = np.zeros(n_arms)
        else:
            self.q_values = np.array(initial_values)

        # Track number of times each arm has been selected
        self.arm_counts = np.zeros(n_arms)

        # Total number of actions taken
        self.total_actions = 0

        # Track history for analysis
        self.action_history = []
        self.reward_history = []
        self.cumulative_reward = 0
        self.ucb_values_history = []

    def calculate_ucb_values(self):
        """
        Calculate UCB values for all arms.

        UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))

        For arms that haven't been selected (N(a) = 0), we assign
        infinity to ensure they get selected first.

        Returns:
            ucb_values: UCB values for each arm
        """
        ucb_values = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            if self.arm_counts[arm] == 0:
                # Unselected arms get infinite UCB value
                ucb_values[arm] = float("inf")
            else:
                # UCB formula
                confidence_interval = self.c * math.sqrt(
                    math.log(self.total_actions) / self.arm_counts[arm]
                )
                ucb_values[arm] = self.q_values[arm] + confidence_interval

        return ucb_values

    def select_action(self):
        """
        Select the arm with the highest UCB value.

        Returns:
            action: Selected arm index
        """
        ucb_values = self.calculate_ucb_values()

        # Store UCB values for analysis
        self.ucb_values_history.append(ucb_values.copy())

        # Select arm with highest UCB value
        # Break ties randomly
        max_ucb = np.max(ucb_values)
        best_arms = np.where(ucb_values == max_ucb)[0]
        action = np.random.choice(best_arms)

        return action

    def update_q_value(self, action, reward):
        """
        Update the Q-value for the selected action using incremental average.

        Q_new(a) = Q_old(a) + (1/n) * [R - Q_old(a)]
        where n is the number of times action a has been selected.

        Args:
            action: Selected arm index
            reward: Received reward
        """
        self.arm_counts[action] += 1
        self.total_actions += 1

        # Incremental update rule
        step_size = 1.0 / self.arm_counts[action]
        self.q_values[action] += step_size * (reward - self.q_values[action])

    def run_episode(self, bandit_arms, n_steps):
        """
        Run the UCB algorithm for n_steps.

        Args:
            bandit_arms: List of bandit arms (reward distributions)
            n_steps: Number of steps to run

        Returns:
            total_reward: Total reward accumulated
        """
        total_reward = 0

        for step in range(n_steps):
            # Select action using UCB policy
            action = self.select_action()

            # Get reward from the selected arm
            reward = bandit_arms[action].sample_reward()

            # Update Q-value
            self.update_q_value(action, reward)

            # Track progress
            total_reward += reward
            self.cumulative_reward += reward
            self.action_history.append(action)
            self.reward_history.append(reward)

        return total_reward

    def get_best_arm(self):
        """
        Get the arm with the highest estimated Q-value.

        Returns:
            best_arm: Index of the best arm
        """
        return np.argmax(self.q_values)

    def get_statistics(self):
        """
        Get algorithm statistics for analysis.

        Returns:
            dict: Statistics including Q-values, arm counts, etc.
        """
        return {
            "q_values": self.q_values.copy(),
            "arm_counts": self.arm_counts.copy(),
            "total_reward": self.cumulative_reward,
            "action_history": self.action_history.copy(),
            "reward_history": self.reward_history.copy(),
            "ucb_values_history": self.ucb_values_history.copy(),
            "c_parameter": self.c,
            "total_actions": self.total_actions,
        }
