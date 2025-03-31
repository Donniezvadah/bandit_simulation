# bandits/main.py
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from environments import BernoulliBandit, GaussianBandit
from agents import EpsilonGreedyAgent, UCBAgent, ThompsonSamplingAgent
from utils.seeder import set_seed
from utils.confidence import compute_regret_confidence_intervals

def load_config(config_path='config/config.xml'):
    """Load configuration from XML file."""
    tree = ET.parse(config_path)
    root = tree.getroot()
    
    config = {
        'paths': {
            'plots_dir': root.find('paths/plots_dir').text,
            'agents_dir': root.find('paths/agents_dir').text,
            'environments_dir': root.find('paths/environments_dir').text,
        },
        'simulation': {
            'n_steps': int(root.find('simulation/n_steps').text),
            'n_trials': int(root.find('simulation/n_trials').text),
            'confidence_levels': [float(level.text) for level in root.findall('simulation/confidence_levels/level')]
        },
        'seeds': {
            'numpy': int(root.find('seeds/numpy').text),
            'random': int(root.find('seeds/random').text)
        }
    }
    return config

def get_regret(env, agents, n_steps, n_trials):
    """
    Simulates the multi-armed bandit problem for a given environment and a set of agents.
    """
    scores = OrderedDict({
        agent.name: [0.0 for _ in range(n_steps)] for agent in agents
    })

    for trial in range(n_trials):
        env.reset()

        for agent in agents:
            agent.init_actions(env.action_count)

        for i in range(n_steps):
            optimal_reward = env.optimal_reward()

            for agent in agents:
                action = agent.get_action()
                reward = env.pull(action)
                agent.update(action, reward)
                scores[agent.name][i] += optimal_reward - reward

            env.step()

    for agent in agents:
        scores[agent.name] = np.cumsum(scores[agent.name]) / n_trials

    return scores

def plot_regret_with_confidence(agents, regret, confidence_intervals, config):
    """Plot regret with confidence intervals."""
    plt.figure(figsize=(10, 6))
    
    for agent in agents:
        plt.plot(regret[agent.name], label=agent.name)
        for ci_name, (lower, upper) in confidence_intervals[agent.name].items():
            plt.fill_between(range(len(regret[agent.name])), lower, upper, alpha=0.2)
    
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('Regret with Confidence Intervals')
    plt.legend()
    
    # Save plots
    plt.savefig(f"{config['paths']['plots_dir']}/regret_with_ci.png")
    plt.savefig(f"{config['paths']['plots_dir']}/regret_with_ci.pdf")
    plt.close()

def main():
    # Load configuration
    config = load_config()
    
    # Set random seeds
    set_seed(config['seeds']['numpy'])
    
    # Initialize environments
    bernoulli_env = BernoulliBandit()
    gaussian_env = GaussianBandit()
    
    # Initialize agents
    agents = [
        EpsilonGreedyAgent(),
        UCBAgent(),
        ThompsonSamplingAgent()
    ]
    
    # Run simulations for both environments
    for env_name, env in [('Bernoulli', bernoulli_env), ('Gaussian', gaussian_env)]:
        # Get regret scores
        regret = get_regret(env, agents, config['simulation']['n_steps'], config['simulation']['n_trials'])
        
        # Compute confidence intervals
        confidence_intervals = compute_regret_confidence_intervals(
            regret, 
            config['simulation']['confidence_levels']
        )
        
        # Plot results
        plot_regret_with_confidence(agents, regret, confidence_intervals, config)

if __name__ == "__main__":
    main()

# Add seeder for the bandits algorithms
#Now do a confidence interval 99% for the regret
# Create a folder called agents and save the agents in there and their strategies , and __init__.py file to import the agents add base agent  
# create a folder for the plots and save the plots in there
# folder for configuration and save the configuration in there 

