# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from q_learning_ucb import QLearningAgent
from market import MarketEnvironment

import random
import itertools
import time
import copy

UCB = False
np.random.seed(999)

# Define the hyperparameter search space
if UCB:
    Delta_values = np.linspace(0.1, 1.0, num=3)  # Creates 10 values from 0.1 to 1.0 with a spacing of 0.1
    bonus_coef_0_values = np.linspace(0.1, 1.0, num=3)
    bonus_coef_1_values = [0.5, 1.0]
    ucb_H_values = [5,10]
    Q_upper_bound_values = np.linspace(3.0, 4.0, num=3)
    # Define parameter names
    parameter_names = ['Delta', 'bonus_coef_0', 'bonus_coef_1', 'ucb_H', 'Q_upper_bound']

else:
    Delta_values = [0.1]
    eps0_values = [0.9, 0.95, 0.99]
    exp0_values = [0.7, 0.8, 0.9]

    # Define parameter names
    parameter_names = ['Delta', 'eps0', 'exp0']


# Create a list of all possible hyperparameter combinations
vars = [eval(param + '_values') for param in parameter_names]
hyperparameter_combinations = itertools.product(*vars)

# Initialize a dictionary to store results
results = {}

dim_price_grid = 10 # N_P: price grid dimension - 1 (because we start from 0)
bound_inventory = 5 # N_Y: (inventory grid dimension - 1)/2 (because we allow both - and + and 0)

dim_midprice_grid = 2*dim_price_grid-1
dim_inventory_grid = 2*bound_inventory+1
dim_action_ask_price = dim_price_grid+2
dim_action_buy_price = dim_price_grid+2

# Define a function to evaluate the agent with given hyperparameters
def evaluate_agent(**kwargs):
    env = MarketEnvironment(dim_price_grid, bound_inventory, dim_action_ask_price, dim_action_buy_price, kwargs['Delta'])
    env.reset()

    agent = QLearningAgent(env, dim_midprice_grid, dim_inventory_grid, dim_action_ask_price, dim_action_buy_price,
                            UCB=UCB, N_RL_iter=10**5, **kwargs)
    start_time = time.time()
    agent.update()
    end_time = time.time()
    result = agent.results_check()
    return result, end_time - start_time

# Progress counter
total_combinations = len(list(copy.copy(hyperparameter_combinations)))
print(f"Total number of combinations: {total_combinations}")
progress = 0

# Perform grid search
for vars in hyperparameter_combinations:
    print(vars)
    # Record parameter names and values in the result
    param_values = vars
    param_dict = {parameter_names[i]: param_values[i] for i in range(len(parameter_names))}
    result, runtime = evaluate_agent(**param_dict)

    result_with_params = {'parameters': param_dict, 'result': result, 'runtime': runtime}
        
    # Print progress
    progress += 1
    print(f"Progress: {progress}/{total_combinations}")
    print("Hyperparameters:")
    for param_name, param_value in param_dict.items():
        print(f"{param_name}: {param_value}")
    
    print(f"Runtime: {runtime:.4f} seconds")
    
    results[tuple(param_values)] = result_with_params

# Save results to a file
if UCB:
    file_path = f"results/hyperparameter_results_UCB_NP{dim_price_grid}_NY{bound_inventory}.txt"
else:
    file_path = f"results/hyperparameter_results_NP{dim_price_grid}_NY{bound_inventory}.txt"
with open(file_path, "w") as f:
    for params, result_info in results.items():
        f.write(f"Hyperparameters:\n")
        for param_name, param_value in result_info['parameters'].items():
            f.write(f"{param_name}: {param_value}\n")
        f.write(f"Result: {result_info['result']}\n")
        f.write(f"Runtime: {result_info['runtime']} seconds\n")
        f.write("\n")