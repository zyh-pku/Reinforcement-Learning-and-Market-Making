KKK = 0

import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import copy

import os

new_dir = "/mnt/gsmfs2/home/libuser27/A_RLMM_2"

os.chdir(new_dir)

from Nash_q_learning import NashQLearningAgent
from market_2 import MarketEnvironment

Delta_values = [ np.logspace(-1, -2, num=10)[KKK] ]

exp0_values = [0.9] #np.linspace(0.25, 0.95, num=5)
exp_values = [0.7] #[0.3, 0.5, 0.65, 0.8, 0.95] # [0.8, 1.0]
exp_epoch_values = [5]#[5,10, 25, 50 ]

lr0_values = [0.5]
lr_values = [0.8]
lr_epoch_values = [10]
# exp0_values = [0.7, 0.8, 0.9]
# eps0_values = np.linspace(0.1, 0.9, num=9)

# Define parameter names
parameter_names = ['Delta',  'lr0', 'lr', 'lr_epoch', 'exp0', 'exp', 'exp_epoch' ]


# Create a list of all possible hyperparameter combinations
vars = [eval(param + '_values') for param in parameter_names]
hyperparameter_combinations = itertools.product(*vars)

# Initialize a dictionary to store results
results = {}

dim_price_grid = 2 # N_P: price grid dimension - 1 (because we start from 0)

# Define a function to evaluate the agent with given hyperparameters
def evaluate_agent(**kwargs):
    env = MarketEnvironment(dim_price_grid, kwargs['Delta'])
    env.reset()
    agent = NashQLearningAgent(env, dim_price_grid,  N_learning_steps=5, N_RL_iter=5, V_RL_iter_initial = 3.75, **kwargs)
    start_time = time.time()
    np.random.seed(999)
    agent.update()
    end_time = time.time()
    V_error_1, V_error_2, pi_error_1, pi_error_2 = agent.result_metrics()
    return V_error_1, V_error_2, pi_error_1, pi_error_2, end_time - start_time


def write_to_file(file_path, results):
    with open(file_path, "w") as f:
        for params, result_info in results.items():
            f.write(f"Hyperparameters:\n")
            for param_name, param_value in result_info['parameters'].items():
                f.write(f"{param_name}: {param_value}\n")
            f.write(f"V_error_1: {result_info['V_error_1']}\n")
            f.write(f"V_error_2: {result_info['V_error_2']}\n")
            f.write(f"pi_error_1: {result_info['pi_error_1']}\n")
            f.write(f"pi_error_2: {result_info['pi_error_2']}\n")
            f.write(f"Runtime: {result_info['runtime']} seconds\n")
            f.write("\n")


file_path = f"results/hyperparameter_results_NP{dim_price_grid}_Delta{KKK}.txt"


# Perform grid search
for vars in hyperparameter_combinations:
    # print(vars)
    # Record parameter names and values in the result
    param_values = vars
    param_dict = {parameter_names[i]: param_values[i] for i in range(len(parameter_names))}
    V_error_1, V_error_2, pi_error_1, pi_error_2, runtime = evaluate_agent(**param_dict) #new add
    if pi_error_1>-1 and pi_error_2>-1 : # pi_error_1<0.5 and pi_error_2<0.5:
        result_with_params = {'parameters': param_dict, 
                            'V_error_1':V_error_1, 'V_error_2':V_error_2, 
                            'pi_error_1':pi_error_1, 'pi_error_2':pi_error_2,
                            'runtime': runtime} #new add
        # Print progress
        # progress += 1
        # print(f"Progress: {progress}/{total_combinations}")
        # print("Hyperparameters:")
        # for param_name, param_value in param_dict.items():
        #     print(f"{param_name}: {param_value}")
        # print(f"Runtime: {runtime:.4f} seconds")
        results[tuple(param_values)] = result_with_params
        # Write results to a file
        write_to_file(file_path, results)