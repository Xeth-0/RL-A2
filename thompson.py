import numpy as np
import random

class ThompsonSampling:
    """
    Thompson Sampling algorithm for Bernoulli multi-armed bandit problems.
    
    Thompson Sampling uses Bayesian inference to maintain a probability
    distribution over the expected reward of each arm. It samples from
    these distributions and selects the arm with the highest sample.
    
    For Bernoulli bandits, we use Beta distributions as conjugate priors:
    - Beta(α, β) represents our belief about the success probability
    - α represents successes + 1 (prior successes)
    - β represents failures + 1 (prior failures)
    
    The algorithm naturally balances exploration and exploitation through
    the uncertainty encoded in the posterior distributions.
    """
    
    def __init__(self, n_arms, alpha_prior=1.0, beta_prior=1.0):
        """
        Initialize the Thompson Sampling bandit algorithm.
        
        Args:
            n_arms: Number of bandit arms
            alpha_prior: Prior successes for Beta distribution (default: 1.0)
            beta_prior: Prior failures for Beta distribution (default: 1.0)
        """
        self.n_arms = n_arms
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Initialize Beta distribution parameters
        # Alpha: successes + prior
        # Beta: failures + prior
        self.alphas = np.full(n_arms, alpha_prior)
        self.betas = np.full(n_arms, beta_prior)
        
        # Track number of times each arm has been selected
        self.arm_counts = np.zeros(n_arms)
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)
        
        # Track history for analysis
        self.action_history = []
        self.reward_history = []
        self.cumulative_reward = 0
        self.sampled_values_history = []
        
    def sample_theta(self):
        """
        Sample success probabilities from Beta posterior distributions.
        
        For each arm, sample θ_i ~ Beta(α_i, β_i)
        where α_i and β_i are updated based on observed successes and failures.
        
        Returns:
            theta_samples: Sampled success probabilities for each arm
        """
        theta_samples = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            # Sample from Beta(alpha, beta) distribution
            theta_samples[arm] = np.random.beta(self.alphas[arm], self.betas[arm])
        
        return theta_samples
    
    def select_action(self):
        """
        Select the arm with the highest sampled success probability.
        
        Returns:
            action: Selected arm index
        """
        # Sample success probabilities from posterior distributions
        theta_samples = self.sample_theta()
        
        # Store sampled values for analysis
        self.sampled_values_history.append(theta_samples.copy())
        
        # Select arm with highest sampled probability
        action = np.argmax(theta_samples)
        
        return action
    
    def update_posterior(self, action, reward):
        """
        Update the Beta posterior distribution for the selected arm.
        
        For Bernoulli rewards:
        - If reward = 1 (success): α += 1
        - If reward = 0 (failure): β += 1
        
        Args:
            action: Selected arm index
            reward: Received reward (0 or 1 for Bernoulli)
        """
        self.arm_counts[action] += 1
        
        if reward == 1:
            # Success: increment alpha
            self.alphas[action] += 1
            self.successes[action] += 1
        else:
            # Failure: increment beta
            self.betas[action] += 1
            self.failures[action] += 1
    
    def run_episode(self, bandit_arms, n_steps):
        """
        Run the Thompson Sampling algorithm for n_steps.
        
        Args:
            bandit_arms: List of Bernoulli bandit arms
            n_steps: Number of steps to run
            
        Returns:
            total_reward: Total reward accumulated
        """
        total_reward = 0
        
        for step in range(n_steps):
            # Select action using Thompson Sampling
            action = self.select_action()
            
            # Get reward from the selected arm (0 or 1)
            reward = bandit_arms[action].sample_reward()
            
            # Update posterior distribution
            self.update_posterior(action, reward)
            
            # Track progress
            total_reward += reward
            self.cumulative_reward += reward
            self.action_history.append(action)
            self.reward_history.append(reward)
        
        return total_reward
    
    def get_mean_estimates(self):
        """
        Get the posterior mean estimates for each arm.
        
        For Beta(α, β), the mean is α / (α + β)
        
        Returns:
            means: Posterior mean estimates
        """
        return self.alphas / (self.alphas + self.betas)
    
    def get_best_arm(self):
        """
        Get the arm with the highest posterior mean estimate.
        
        Returns:
            best_arm: Index of the best arm
        """
        means = self.get_mean_estimates()
        return np.argmax(means)
    
    def get_statistics(self):
        """
        Get algorithm statistics for analysis.
        
        Returns:
            dict: Statistics including posterior parameters, arm counts, etc.
        """
        return {
            'alphas': self.alphas.copy(),
            'betas': self.betas.copy(),
            'mean_estimates': self.get_mean_estimates(),
            'arm_counts': self.arm_counts.copy(),
            'successes': self.successes.copy(),
            'failures': self.failures.copy(),
            'total_reward': self.cumulative_reward,
            'action_history': self.action_history.copy(),
            'reward_history': self.reward_history.copy(),
            'sampled_values_history': self.sampled_values_history.copy()
        }

class BernoulliBanditArm:
    """
    Bernoulli bandit arm with a specific success probability.
    """
    
    def __init__(self, success_prob):
        """
        Initialize a Bernoulli bandit arm.
        
        Args:
            success_prob: True success probability (0 ≤ p ≤ 1)
        """
        self.success_prob = success_prob
        
    def sample_reward(self):
        """
        Sample a reward from the Bernoulli distribution.
        
        Returns:
            reward: 1 with probability p, 0 with probability 1-p
        """
        return np.random.binomial(1, self.success_prob)

def run_thompson_sampling_experiment(arm_probs, n_steps=1000, n_runs=100):
    """
    Run Thompson Sampling experiments with Bernoulli bandits.
    
    Args:
        arm_probs: List of true success probabilities for each arm
        n_steps: Number of steps per run
        n_runs: Number of independent runs for averaging
        
    Returns:
        results: Dictionary containing experimental results
    """
    results = {}
    optimal_arm = np.argmax(arm_probs)
    optimal_prob = arm_probs[optimal_arm]
    
    print(f"Running Thompson Sampling Experiment")
    print(f"Arm probabilities: {arm_probs}")
    print(f"Optimal arm: {optimal_arm} (probability: {optimal_prob:.3f})")
    print("-" * 50)
    
    total_rewards = []
    cumulative_rewards = np.zeros(n_steps)
    cumulative_regrets = np.zeros(n_steps)
    arm_selections = np.zeros(len(arm_probs))
    
    for run in range(n_runs):
        # Create Bernoulli bandit arms
        bandit_arms = [BernoulliBanditArm(prob) for prob in arm_probs]
        
        # Initialize algorithm
        agent = ThompsonSampling(len(arm_probs))
        
        # Run episode
        run_reward = 0
        for step in range(n_steps):
            action = agent.select_action()
            reward = bandit_arms[action].sample_reward()
            agent.update_posterior(action, reward)
            
            run_reward += reward
            cumulative_rewards[step] += reward
            
            # Calculate regret (difference from optimal expected reward)
            regret = optimal_prob - reward
            cumulative_regrets[step] += regret
            
            arm_selections[action] += 1
        
        total_rewards.append(run_reward)
    
    # Average over all runs
    avg_cumulative_rewards = cumulative_rewards / n_runs
    avg_cumulative_regrets = np.cumsum(cumulative_regrets) / n_runs
    avg_arm_selections = arm_selections / (n_runs * n_steps)
    
    results = {
        'avg_total_reward': np.mean(total_rewards),
        'std_total_reward': np.std(total_rewards),
        'cumulative_rewards': avg_cumulative_rewards,
        'cumulative_regrets': avg_cumulative_regrets,
        'arm_selection_freq': avg_arm_selections,
        'final_regret': avg_cumulative_regrets[-1]
    }
    
    print(f"Average total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Final cumulative regret: {avg_cumulative_regrets[-1]:.2f}")
    print(f"Success rate: {np.mean(total_rewards) / n_steps:.3f}")
    
    return results

def compare_all_algorithms(arm_probs, n_steps=1000, n_runs=100):
    """
    Compare Thompson Sampling with UCB and Epsilon-Greedy on Bernoulli bandits.
    
    Args:
        arm_probs: List of true success probabilities for each arm
        n_steps: Number of steps per run
        n_runs: Number of independent runs for averaging
        
    Returns:
        results: Comparison results
    """
    from ucb import UCB
    from e_greedy import EpsilonGreedy, BanditArm
    
    results = {}
    optimal_arm = np.argmax(arm_probs)
    optimal_prob = arm_probs[optimal_arm]
    
    print(f"Comparing All Algorithms on Bernoulli Bandits")
    print(f"Arm probabilities: {arm_probs}")
    print(f"Optimal arm: {optimal_arm} (probability: {optimal_prob:.3f})")
    print("-" * 50)
    
    # Test configurations
    algorithms = {
        'Thompson Sampling': lambda n_arms: ThompsonSampling(n_arms),
        'UCB (c=2.0)': lambda n_arms: UCB(n_arms, c=2.0),
        'ε-greedy (ε=0.1)': lambda n_arms: EpsilonGreedy(n_arms, epsilon=0.1),
        'ε-greedy (ε=0.05)': lambda n_arms: EpsilonGreedy(n_arms, epsilon=0.05)
    }
    
    for alg_name, alg_constructor in algorithms.items():
        print(f"Testing {alg_name}")
        
        total_rewards = []
        cumulative_rewards = np.zeros(n_steps)
        cumulative_regrets = np.zeros(n_steps)
        arm_selections = np.zeros(len(arm_probs))
        
        for run in range(n_runs):
            # Create bandit arms (use Bernoulli for Thompson Sampling comparison)
            if alg_name == 'Thompson Sampling':
                bandit_arms = [BernoulliBanditArm(prob) for prob in arm_probs]
                agent = alg_constructor(len(arm_probs))
                
                # Special handling for Thompson Sampling
                run_reward = 0
                for step in range(n_steps):
                    action = agent.select_action()
                    reward = bandit_arms[action].sample_reward()
                    agent.update_posterior(action, reward)
                    
                    run_reward += reward
                    cumulative_rewards[step] += reward
                    
                    regret = optimal_prob - reward
                    cumulative_regrets[step] += regret
                    
                    arm_selections[action] += 1
            else:
                # Use Bernoulli arms for other algorithms too
                bandit_arms = [BanditArm(prob, distribution='bernoulli') for prob in arm_probs]
                agent = alg_constructor(len(arm_probs))
                
                run_reward = 0
                for step in range(n_steps):
                    action = agent.select_action()
                    reward = bandit_arms[action].sample_reward()
                    agent.update_q_value(action, reward)
                    
                    run_reward += reward
                    cumulative_rewards[step] += reward
                    
                    regret = optimal_prob - reward
                    cumulative_regrets[step] += regret
                    
                    arm_selections[action] += 1
            
            total_rewards.append(run_reward)
        
        # Average over all runs
        avg_cumulative_rewards = cumulative_rewards / n_runs
        avg_cumulative_regrets = np.cumsum(cumulative_regrets) / n_runs
        avg_arm_selections = arm_selections / (n_runs * n_steps)
        
        results[alg_name] = {
            'avg_total_reward': np.mean(total_rewards),
            'std_total_reward': np.std(total_rewards),
            'cumulative_rewards': avg_cumulative_rewards,
            'cumulative_regrets': avg_cumulative_regrets,
            'arm_selection_freq': avg_arm_selections,
            'final_regret': avg_cumulative_regrets[-1]
        }
        
        print(f"  Average total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"  Success rate: {np.mean(total_rewards) / n_steps:.3f}")
        print(f"  Final cumulative regret: {avg_cumulative_regrets[-1]:.2f}")
    
    return results
