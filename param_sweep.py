# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import time
import copy
import argparse
import os

# new_dir = "/mnt/gsmfs2/home/libuser27/A_RLMM"
# os.chdir(new_dir)

from q_learning_ucb import QLearningAgent
from market import MarketEnvironment

UCB = False

# Delta_index = 1
# Delta_values = [ np.logspace(-1, -3, num=10)[Delta_index] ]

# parse arguments delta
parser = argparse.ArgumentParser()
parser.add_argument('--delta_idx', type=int, default=0)
args = parser.parse_args()
Delta_values = [args.delta_idx]

# Define the hyperparameter search space
# if UCB:
#     Delta_values = np.linspace(0.1, 1.0, num=3)  # Creates 10 values from 0.1 to 1.0 with a spacing of 0.1
#     bonus_coef_0_values = np.linspace(0.1, 1.0, num=3)
#     bonus_coef_1_values = [0.5, 1.0]
#     ucb_H_values = [5,10]
#     Q_upper_bound_values = np.linspace(3.0, 4.0, num=3)
#     # Define parameter names
#     parameter_names = ['Delta', 'bonus_coef_0', 'bonus_coef_1', 'ucb_H', 'Q_upper_bound']
# else: 
# # the following are a quick test:
# lr_exponent_values = np.linspace(0.501, 0.9, num=5)[0:1]
# lr_values = [0.25, 0.5, 0.75, 1.0][0:1]

# exp0_values = np.linspace(0.25, 0.95, num=5)[0:1]
# exp_epoch_values = [5, 10, 25, 50 ][0:1]
# exp_values = [0.3, 0.5, 0.65, 0.8, 0.95][0:1] 
# # the following are the full test:
lr_exponent_values = np.linspace(0.501, 0.9, num=4)
lr_values = [0.1, 0.5, 1.0]
exp0_values = np.linspace(0.25, 0.95, num=4)
exp_epoch_values = [5, 10, 25, 50 ]
exp_values = [0.3, 0.5, 0.7, 0.9]

# Define parameter names
parameter_names = ['Delta', 'lr', 'lr_exponent', 'exp0', 'exp_epoch', 'exp' ]

# stop the RL iteration when the value function error is less than this threshold:
V_error_threshold_for_RL_iteration_stop = 0.075 


# Create a list of all possible hyperparameter combinations
vars = [eval(param + '_values') for param in parameter_names]
hyperparameter_combinations = itertools.product(*vars)

# Initialize a dictionary to store results
results = {}

dim_price_grid = 2 # N_P: price grid dimension - 1 (because we start from 0)
bound_inventory = 1 # N_Y: (inventory grid dimension - 1)/2 (because we allow both - and + and 0)

dim_midprice_grid = 2*dim_price_grid-1
dim_inventory_grid = 2*bound_inventory+1
dim_action_ask_price = dim_price_grid+2
dim_action_buy_price = dim_price_grid+2

# Define a function to evaluate the agent with given hyperparameters
def evaluate_agent(**kwargs):
    env = MarketEnvironment(dim_price_grid, bound_inventory, dim_action_ask_price, dim_action_buy_price, kwargs['Delta'])
    env.reset()
    agent = QLearningAgent(env, dim_midprice_grid, dim_inventory_grid, dim_action_ask_price, dim_action_buy_price,
                           V_RL_iter_threshold = V_error_threshold_for_RL_iteration_stop, 
                           V_RL_iter_initial = 2.5, N_RL_iter=2*10**4, N_learning_steps=2*10**4,
                            UCB=UCB, **kwargs)
    start_time = time.time()
    np.random.seed(999)
    agent.update()
    end_time = time.time()
    policy_error = agent.result_metrics()
    value_error = agent.V_error # value function error
    RL_iter_steps = agent.V_RL_iter_steps # it is the number of RL iterations such that the value function error is less than the threshold V_error_threshold_for_RL_iteration_stop
    # RL_iter_steps will be N_RL_iter if the value function error is not less than the threshold after N_RL_iter iterations
    return policy_error, value_error, RL_iter_steps, end_time - start_time


def write_to_file(file_path, results):
    with open(file_path, "w") as f:
        for params, result_info in results.items():
            f.write(f"Hyperparameters:\n")
            for param_name, param_value in result_info['parameters'].items():
                f.write(f"{param_name}: {param_value}\n")
            f.write(f"Policy_Error: {result_info['policy_error']}\n")
            f.write(f"Value_Error: {result_info['value_error']}\n")
            f.write(f"RL_Iteration_Steps: {result_info['RL_iter_steps']}\n")
            f.write(f"Runtime: {result_info['runtime']} seconds\n")
            f.write("\n")

# Progress counter
total_combinations = len(list(copy.copy(hyperparameter_combinations)))
print(f"Total number of combinations: {total_combinations}")
progress = 0

# Save results to a file
# if UCB:
#     file_path = f"results/hyperparameter_results_UCB_NP{dim_price_grid}_NY{bound_inventory}.txt"
# else:
file_path = f"results/hyperparameter_results_NP{dim_price_grid}_NY{bound_inventory}_Delta{args.delta_idx}.txt"


# Perform grid search
for vars in hyperparameter_combinations:
    print(vars)
    # Record parameter names and values in the result
    param_values = vars
    param_dict = {parameter_names[i]: param_values[i] for i in range(len(parameter_names))}
    policy_error, value_error, RL_iter_steps, runtime = evaluate_agent(**param_dict) #new add
    result_with_params = {'parameters': param_dict, 'policy_error': policy_error, 'value_error': value_error, 'RL_iter_steps': RL_iter_steps, 'runtime': runtime} #new add
    # Print progress
    progress += 1
    print(f"Progress: {progress}/{total_combinations}")
    print("Hyperparameters:")
    for param_name, param_value in param_dict.items():
        print(f"{param_name}: {param_value}")
    print(f"Runtime: {runtime:.4f} seconds")
    results[tuple(param_values)] = result_with_params
    # Write results to a file
    write_to_file(file_path, results)