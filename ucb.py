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
                ucb_values[arm] = float('inf')
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
            'q_values': self.q_values.copy(),
            'arm_counts': self.arm_counts.copy(),
            'total_reward': self.cumulative_reward,
            'action_history': self.action_history.copy(),
            'reward_history': self.reward_history.copy(),
            'ucb_values_history': self.ucb_values_history.copy(),
            'c_parameter': self.c,
            'total_actions': self.total_actions
        }

def run_ucb_experiment(arm_means, c_values, n_steps=1000, n_runs=100):
    """
    Run UCB experiments with different exploration parameters.
    
    Args:
        arm_means: List of true mean rewards for each arm
        c_values: List of c parameter values to test
        n_steps: Number of steps per run
        n_runs: Number of independent runs for averaging
        
    Returns:
        results: Dictionary containing experimental results
    """
    from e_greedy import BanditArm  # Import BanditArm from epsilon-greedy
    
    results = {}
    optimal_arm = np.argmax(arm_means)
    optimal_reward = arm_means[optimal_arm]
    
    print(f"Running UCB Experiment")
    print(f"Arm means: {arm_means}")
    print(f"Optimal arm: {optimal_arm} (reward: {optimal_reward:.2f})")
    print(f"C values: {c_values}")
    print("-" * 50)
    
    for c in c_values:
        print(f"Testing c = {c}")
        
        total_rewards = []
        cumulative_rewards = np.zeros(n_steps)
        cumulative_regrets = np.zeros(n_steps)
        arm_selections = np.zeros(len(arm_means))
        
        for run in range(n_runs):
            # Create bandit arms
            bandit_arms = [BanditArm(mean) for mean in arm_means]
            
            # Initialize algorithm
            agent = UCB(len(arm_means), c=c)
            
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
        
        results[c] = {
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

def compare_ucb_epsilon_greedy(arm_means, n_steps=1000, n_runs=100):
    """
    Compare UCB with Epsilon-Greedy using optimized parameters.
    
    Args:
        arm_means: List of true mean rewards for each arm
        n_steps: Number of steps per run
        n_runs: Number of independent runs for averaging
        
    Returns:
        results: Comparison results
    """
    from e_greedy import EpsilonGreedy, BanditArm
    
    results = {}
    optimal_arm = np.argmax(arm_means)
    optimal_reward = arm_means[optimal_arm]
    
    print(f"Comparing UCB vs Epsilon-Greedy")
    print(f"Arm means: {arm_means}")
    print(f"Optimal arm: {optimal_arm} (reward: {optimal_reward:.2f})")
    print("-" * 50)
    
    # Test configurations
    algorithms = {
        'UCB (c=2.0)': lambda n_arms: UCB(n_arms, c=2.0),
        'UCB (c=1.0)': lambda n_arms: UCB(n_arms, c=1.0),
        'ε-greedy (ε=0.1)': lambda n_arms: EpsilonGreedy(n_arms, epsilon=0.1),
        'ε-greedy (ε=0.01)': lambda n_arms: EpsilonGreedy(n_arms, epsilon=0.01)
    }
    
    for alg_name, alg_constructor in algorithms.items():
        print(f"Testing {alg_name}")
        
        total_rewards = []
        cumulative_rewards = np.zeros(n_steps)
        cumulative_regrets = np.zeros(n_steps)
        arm_selections = np.zeros(len(arm_means))
        
        for run in range(n_runs):
            # Create bandit arms
            bandit_arms = [BanditArm(mean) for mean in arm_means]
            
            # Initialize algorithm
            agent = alg_constructor(len(arm_means))
            
            # Run episode
            run_reward = 0
            for step in range(n_steps):
                action = agent.select_action()
                reward = bandit_arms[action].sample_reward()
                agent.update_q_value(action, reward)
                
                run_reward += reward
                cumulative_rewards[step] += reward
                
                # Calculate regret
                regret = optimal_reward - reward
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
        print(f"  Final cumulative regret: {avg_cumulative_regrets[-1]:.2f}")
    
    return results
