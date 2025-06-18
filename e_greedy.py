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
            'q_values': self.q_values.copy(),
            'arm_counts': self.arm_counts.copy(),
            'total_reward': self.cumulative_reward,
            'action_history': self.action_history.copy(),
            'reward_history': self.reward_history.copy(),
            'epsilon': self.epsilon
        }

class BanditArm:
    """
    Individual bandit arm with a specific reward distribution.
    """
    
    def __init__(self, mean, std=1.0, distribution='normal'):
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
        if self.distribution == 'normal':
            return np.random.normal(self.mean, self.std)
        elif self.distribution == 'bernoulli':
            return np.random.binomial(1, self.mean)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

def run_epsilon_greedy_experiment(arm_means, epsilon_values, n_steps=1000, n_runs=100):
    """
    Run epsilon-greedy experiments with different epsilon values.
    
    Args:
        arm_means: List of true mean rewards for each arm
        epsilon_values: List of epsilon values to test
        n_steps: Number of steps per run
        n_runs: Number of independent runs for averaging
        
    Returns:
        results: Dictionary containing experimental results
    """
    results = {}
    optimal_arm = np.argmax(arm_means)
    optimal_reward = arm_means[optimal_arm]
    
    print(f"Running Epsilon-Greedy Experiment")
    print(f"Arm means: {arm_means}")
    print(f"Optimal arm: {optimal_arm} (reward: {optimal_reward:.2f})")
    print(f"Epsilon values: {epsilon_values}")
    print("-" * 50)
    
    for epsilon in epsilon_values:
        print(f"Testing ε = {epsilon}")
        
        total_rewards = []
        cumulative_rewards = np.zeros(n_steps)
        cumulative_regrets = np.zeros(n_steps)
        arm_selections = np.zeros(len(arm_means))
        
        for run in range(n_runs):
            # Create bandit arms
            bandit_arms = [BanditArm(mean) for mean in arm_means]
            
            # Initialize algorithm
            agent = EpsilonGreedy(len(arm_means), epsilon=epsilon)
            
            # Run episode
            run_reward = 0
            for step in range(n_steps):
                action = agent.select_action()
                reward = bandit_arms[action].sample_reward()
                agent.update_q_value(action, reward)
                
                run_reward += reward
                cumulative_rewards[step] += reward
                
                # Calculate regret (difference from optimal)
                regret = optimal_reward - reward
                cumulative_regrets[step] += regret
                
                arm_selections[action] += 1
            
            total_rewards.append(run_reward)
        
        # Average over all runs
        avg_cumulative_rewards = cumulative_rewards / n_runs
        avg_cumulative_regrets = np.cumsum(cumulative_regrets) / n_runs
        avg_arm_selections = arm_selections / (n_runs * n_steps)
        
        results[epsilon] = {
            'avg_total_reward': np.mean(total_rewards),
            'std_total_reward': np.std(total_rewards),
            'cumulative_rewards': avg_cumulative_rewards,
            'cumulative_regrets': avg_cumulative_regrets,
            'arm_selection_freq': avg_arm_selections,
            'final_regret': avg_cumulative_regrets[-1]
        }
        
        print(f"  Average total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"  Final cumulative regret: {avg_cumulative_regrets[-1]:.2f}")
    
    return results
