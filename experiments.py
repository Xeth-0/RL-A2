import numpy as np
from ucb import UCB
from e_greedy import EpsilonGreedy, BanditArm
from thompson import ThompsonSampling, BernoulliBanditArm


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
            "avg_total_reward": np.mean(total_rewards),
            "std_total_reward": np.std(total_rewards),
            "cumulative_rewards": avg_cumulative_rewards,
            "cumulative_regrets": avg_cumulative_regrets,
            "arm_selection_freq": avg_arm_selections,
            "final_regret": avg_cumulative_regrets[-1],
        }

        print(
            f"  Average total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}"
        )
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
        "UCB (c=2.0)": lambda n_arms: UCB(n_arms, c=2.0),
        "UCB (c=1.0)": lambda n_arms: UCB(n_arms, c=1.0),
        "ε-greedy (ε=0.1)": lambda n_arms: EpsilonGreedy(n_arms, epsilon=0.1),
        "ε-greedy (ε=0.01)": lambda n_arms: EpsilonGreedy(n_arms, epsilon=0.01),
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
            "avg_total_reward": np.mean(total_rewards),
            "std_total_reward": np.std(total_rewards),
            "cumulative_rewards": avg_cumulative_rewards,
            "cumulative_regrets": avg_cumulative_regrets,
            "arm_selection_freq": avg_arm_selections,
            "final_regret": avg_cumulative_regrets[-1],
        }

        print(
            f"  Average total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}"
        )
        print(f"  Final cumulative regret: {avg_cumulative_regrets[-1]:.2f}")

    return results


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
            "avg_total_reward": np.mean(total_rewards),
            "std_total_reward": np.std(total_rewards),
            "cumulative_rewards": avg_cumulative_rewards,
            "cumulative_regrets": avg_cumulative_regrets,
            "arm_selection_freq": avg_arm_selections,
            "final_regret": avg_cumulative_regrets[-1],
        }

        print(
            f"  Average total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}"
        )
        print(f"  Final cumulative regret: {avg_cumulative_regrets[-1]:.2f}")

    return results


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
        "avg_total_reward": np.mean(total_rewards),
        "std_total_reward": np.std(total_rewards),
        "cumulative_rewards": avg_cumulative_rewards,
        "cumulative_regrets": avg_cumulative_regrets,
        "arm_selection_freq": avg_arm_selections,
        "final_regret": avg_cumulative_regrets[-1],
    }

    print(
        f"Average total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}"
    )
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
    results = {}
    optimal_arm = np.argmax(arm_probs)
    optimal_prob = arm_probs[optimal_arm]

    print(f"Comparing All Algorithms on Bernoulli Bandits")
    print(f"Arm probabilities: {arm_probs}")
    print(f"Optimal arm: {optimal_arm} (probability: {optimal_prob:.3f})")
    print("-" * 50)

    # Test configurations
    algorithms = {
        "Thompson Sampling": lambda n_arms: ThompsonSampling(n_arms),
        "UCB (c=2.0)": lambda n_arms: UCB(n_arms, c=2.0),
        "ε-greedy (ε=0.1)": lambda n_arms: EpsilonGreedy(n_arms, epsilon=0.1),
        "ε-greedy (ε=0.05)": lambda n_arms: EpsilonGreedy(n_arms, epsilon=0.05),
    }

    for alg_name, alg_constructor in algorithms.items():
        print(f"Testing {alg_name}")

        total_rewards = []
        cumulative_rewards = np.zeros(n_steps)
        cumulative_regrets = np.zeros(n_steps)
        arm_selections = np.zeros(len(arm_probs))

        for run in range(n_runs):
            # Create bandit arms (use Bernoulli for Thompson Sampling comparison)
            if alg_name == "Thompson Sampling":
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
                bandit_arms = [
                    BanditArm(prob, distribution="bernoulli") for prob in arm_probs
                ]
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
            "avg_total_reward": np.mean(total_rewards),
            "std_total_reward": np.std(total_rewards),
            "cumulative_rewards": avg_cumulative_rewards,
            "cumulative_regrets": avg_cumulative_regrets,
            "arm_selection_freq": avg_arm_selections,
            "final_regret": avg_cumulative_regrets[-1],
        }

        print(
            f"  Average total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}"
        )
        print(f"  Success rate: {np.mean(total_rewards) / n_steps:.3f}")
        print(f"  Final cumulative regret: {avg_cumulative_regrets[-1]:.2f}")

    return results
