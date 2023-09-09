# -*- coding: utf-8 -*-
"""
@author: Niko Zuppas

Description: This script provides a function for Bayesian probability updating
as part of Chapter 2 Exercises 2M1 and 2M2.

Dependencies: NumPy, Matplotlib, Seaborn
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def grid_approximation_bayesian_update(prior_type, data, grid_size=20):
    """
    Perform Bayesian updating for likelihood and prior.
    
    Parameters:
    prior_type (str): The type of prior to use ('uniform' or 'step').
    data (str): A string where 'W' represents a success (Water) and 'L' represents a failure (Land).
    grid_size (int): The number of points in the probability grid. Default is 20.
    
    Returns:
    None: Displays the posterior distribution as a Seaborn plot.
    """
    
    # Count successes and failures in the data
    n_w = data.count('W')
    n_l = data.count('L')
    
    # Define grid of probabilities
    p_grid = np.linspace(0, 1, grid_size)
    
    # Choose the prior based on the prior_type argument
    if prior_type == 'uniform':
        prior = np.ones(grid_size)
    elif prior_type == 'step':
        prior = np.where(p_grid < 0.5, 0, 1)
    else:
        print("Invalid prior_type. Please choose either 'uniform' or 'step'.")
        return

    # Compute likelihood at each value in the grid
    likelihood = (p_grid ** n_w) * ((1 - p_grid) ** n_l)

    # Compute unnormalized posterior
    unnormalized_posterior = likelihood * prior

    # Normalize the posterior distribution
    posterior = unnormalized_posterior / np.sum(unnormalized_posterior)

    # Set Seaborn style for better aesthetics
    sns.set(style="whitegrid", palette="pastel")

    # Create the plot
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=p_grid, y=posterior, marker="o", color="g", linewidth=3)
    plt.xlabel('Probability of Water', fontsize=14)
    plt.ylabel('Posterior Probability', fontsize=14)
    plt.title(f'Posterior Probability of Water based on {data} using a {prior_type} prior', fontsize=16)
    plt.show()

#2M1.1
grid_approximation_bayesian_update('uniform', 'WWW')
#2M1.2
grid_approximation_bayesian_update('uniform', 'WWWL')
#2M1.3
grid_approximation_bayesian_update('uniform', 'LWWLWWW')

#2M2.1
grid_approximation_bayesian_update('step', 'WWW')
#2M2.2
grid_approximation_bayesian_update('step', 'WWWL')
#2M2.3
grid_approximation_bayesian_update('step', 'LWWLWWW')


