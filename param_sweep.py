# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from q_learning_ucb import QLearningAgent
from market import MarketEnvironment

import random
import itertools
import time

# Define the hyperparameter search space
Delta_values = np.linspace(0.1, 1.0, num=3)  # Creates 10 values from 0.1 to 1.0 with a spacing of 0.1
bonus_coef_0_values = np.linspace(0.1, 1.0, num=3)
bonus_coef_1_values = [0.5, 1.0]
ucb_H_values = [5,10]
Q_upper_bound_values = np.linspace(3.0, 4.0, num=3)
# Define parameter names
parameter_names = ['Delta', 'bonus_coef_0', 'bonus_coef_1', 'ucb_H', 'Q_upper_bound']

# Create a list of all possible hyperparameter combinations
hyperparameter_combinations = itertools.product(Delta_values, bonus_coef_0_values, bonus_coef_1_values, ucb_H_values, Q_upper_bound_values)

# Initialize a dictionary to store results
results = {}

dim_price_grid = 2 # N_P: price grid dimension - 1 (because we start from 0)
bound_inventory = 1 # N_Y: (inventory grid dimension - 1)/2 (because we allow both - and + and 0)
Delta = 0.1

dim_midprice_grid = 2*dim_price_grid-1
dim_inventory_grid = 2*bound_inventory+1
dim_action_ask_price = dim_price_grid+2
dim_action_buy_price = dim_price_grid+2


# Define a function to evaluate the agent with given hyperparameters
def evaluate_agent(delta, bonus_coef_0, bonus_coef_1, ucb_H, Q_upper_bound):
    env = MarketEnvironment(dim_price_grid, bound_inventory, dim_action_ask_price, dim_action_buy_price, Delta)
    env.reset()

    agent = QLearningAgent(env, dim_midprice_grid, dim_inventory_grid, dim_action_ask_price, dim_action_buy_price, delta,
                            UCB=True, N_RL_iter=10**5, bonus_coef_0=bonus_coef_0, bonus_coef_1=bonus_coef_1, ucb_H=ucb_H, Q_upper_bound=Q_upper_bound)
    start_time = time.time()
    agent.update()
    end_time = time.time()
    result = agent.results_check()
    return result, end_time - start_time

# Progress counter
total_combinations = len(Delta_values) * len(bonus_coef_0_values) * len(bonus_coef_1_values) * len(ucb_H_values) * len(Q_upper_bound_values)
progress = 0

# Perform grid search
for delta, bonus_coef_0, bonus_coef_1, ucb_H, Q_upper_bound in hyperparameter_combinations:
    result, runtime = evaluate_agent(delta, bonus_coef_0, bonus_coef_1, ucb_H, Q_upper_bound)
    # Record parameter names and values in the result
    param_values = [delta, bonus_coef_0, bonus_coef_1, ucb_H, Q_upper_bound]
    param_dict = {parameter_names[i]: param_values[i] for i in range(len(parameter_names))}
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
with open("hyperparameter_results.txt", "w") as f:
    for params, result_info in results.items():
        f.write(f"Hyperparameters:\n")
        for param_name, param_value in result_info['parameters'].items():
            f.write(f"{param_name}: {param_value}\n")
        f.write(f"Result: {result_info['result']}\n")
        f.write(f"Runtime: {result_info['runtime']} seconds\n")
        f.write("\n")