import numpy as np
import random


class EpsilonGreedy:
    """
    Epsilon-Greedy algorithm for multi-armed bandit problems.

    The algorithm balances exploration and exploitation:
    - With probability ε: explore (choose random arm)
    - With probability 1-ε: exploit (choose arm with highest estimated reward)

    This simple strategy ensures we explore all arms while gradually
    converging to the best arm as we gain more experience.
    """

    def __init__(self, n_arms, epsilon=0.1, initial_values=None):
        """
        Initialize the Epsilon-Greedy bandit algorithm.

        Args:
            n_arms: Number of bandit arms
            epsilon: Exploration probability (0 ≤ ε ≤ 1)
            initial_values: Initial Q-values for each arm (default: zeros)
        """
        self.n_arms = n_arms
        self.epsilon = epsilon

        # Initialize action-value estimates (Q-values)
        if initial_values is None:
            self.q_values = np.zeros(n_arms)
        else:
            self.q_values = np.array(initial_values)

        # Track number of times each arm has been selected
        self.arm_counts = np.zeros(n_arms)

        # Track history for analysis
        self.action_history = []
        self.reward_history = []
        self.cumulative_reward = 0

    def select_action(self):
        """
        Select an action using epsilon-greedy policy.

        Returns:
            action: Selected arm index
        """
        if random.random() < self.epsilon:
            # Exploration: choose random arm
            action = random.randint(0, self.n_arms - 1)
        else:
            # Exploitation: choose arm with highest Q-value
            # Break ties randomly
            max_q = np.max(self.q_values)
            best_arms = np.where(self.q_values == max_q)[0]
            action = random.choice(best_arms)

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

        # Incremental update rule
        step_size = 1.0 / self.arm_counts[action]
        self.q_values[action] += step_size * (reward - self.q_values[action])

    def run_episode(self, bandit_arms, n_steps):
        """
        Run the epsilon-greedy algorithm for n_steps.

        Args:
            bandit_arms: List of bandit arms (reward distributions)
            n_steps: Number of steps to run

        Returns:
            total_reward: Total reward accumulated
        """
        total_reward = 0

        for step in range(n_steps):
            # Select action using epsilon-greedy policy
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
            "epsilon": self.epsilon,
        }


class BanditArm:
    """
    Individual bandit arm with a specific reward distribution.
    """

    def __init__(self, mean, std=1.0, distribution="normal"):
        """
        Initialize a bandit arm.

        Args:
            mean: Mean reward of the arm
            std: Standard deviation (for normal distribution)
            distribution: Type of distribution ('normal', 'bernoulli')
        """
        self.mean = mean
        self.std = std
        self.distribution = distribution

    def sample_reward(self):
        """
        Sample a reward from the arm's distribution.

        Returns:
            reward: Sampled reward value
        """
        if self.distribution == "normal":
            return np.random.normal(self.mean, self.std)
        elif self.distribution == "bernoulli":
            return np.random.binomial(1, self.mean)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
