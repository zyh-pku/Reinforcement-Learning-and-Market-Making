import matplotlib.pyplot as plt
import re
import pandas as pd
import seaborn as sns

# Function to extract hyperparameters and results from the text
def extract_hyperparameters_and_results(text):
    hyperparameters = {}
    result = None
    runtime = None
    lines = text.strip().split('\n')
    for line in lines:
        if line.startswith("Delta:"):
            hyperparameters["Delta"] = float(line.split(":")[1].strip())
        elif line.startswith("bonus_coef_0:"):
            hyperparameters["bonus_coef_0"] = float(line.split(":")[1].strip())
        elif line.startswith("bonus_coef_1:"):
            hyperparameters["bonus_coef_1"] = float(line.split(":")[1].strip())
        elif line.startswith("ucb_H:"):
            hyperparameters["ucb_H"] = int(line.split(":")[1].strip())
        elif line.startswith("Q_upper_bound:"):
            hyperparameters["Q_upper_bound"] = float(line.split(":")[1].strip())
        elif line.startswith("Result:"):
            result = int(line.split(":")[1].strip())
        elif line.startswith("Runtime:"):
            runtime = float(re.search(r'\d+\.\d+', line).group())  # Extract runtime as a float
    return hyperparameters, result, runtime

def parse_experiment_data_v2(file_lines):
    data = []
    current_experiment = {}
    for line in file_lines:
        # Identifying hyperparameter lines and the result line
        if ':' in line and 'Hyperparameters' not in line:
            key, value = line.split(':')
            # For runtime, remove 'seconds' from the value
            if 'seconds' in value:
                value = value.replace('seconds', '').strip()
            current_experiment[key.strip()] = float(value.strip())
        # When a blank line is encountered, it signifies the end of an experiment's data
        elif line == '\n':
            if current_experiment:  # Ensure the current experiment has data
                data.append(current_experiment)
                current_experiment = {}
    return pd.DataFrame(data)


def plot_param_effects(file_path):
    with open(file_path, 'r') as file:
        file_content = file.readlines()

    # Re-parsing the file content
    df = parse_experiment_data_v2(file_content)

    df.head()


    # Setting the aesthetics for the plots
    sns.set(style="whitegrid")

    # Creating plots for each hyperparameter against the result
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
    fig.suptitle('Effect of Hyperparameters on Experiment Result', fontsize=16)

    # Plotting
    sns.scatterplot(data=df, x='Delta', y='Result', ax=axes[0, 0])
    sns.scatterplot(data=df, x='bonus_coef_0', y='Result', ax=axes[0, 1])
    sns.scatterplot(data=df, x='bonus_coef_1', y='Result', ax=axes[1, 0])
    sns.scatterplot(data=df, x='ucb_H', y='Result', ax=axes[1, 1])
    sns.scatterplot(data=df, x='Q_upper_bound', y='Result', ax=axes[2, 0])

    # Removing the empty subplot
    fig.delaxes(axes[2][1])

    # Adjusting layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    plt.show()


def result_analysis(file_path, save_to):

    # Read the contents of the text file
    with open(file_path, "r") as file:
        lines = file.read()

    plot_param_effects(file_path)

    # Split the text into individual result entries
    result_entries = re.split(r'\n(?=Hyperparameters:)', lines)

    # Initialize lists to store filtered data and all parameter combinations with Result=0
    filtered_data = []
    parameter_combinations_with_result_0 = []

    # Get minimum result
    results = []
    for entry in result_entries:
        hyperparameters, result, runtime = extract_hyperparameters_and_results(entry)
        results.append(result)
    min_result = min(results)

    # Parse the data and filter based on Result
    for entry in result_entries:
        hyperparameters, result, runtime = extract_hyperparameters_and_results(entry)
        if result == min_result:  # or 0
            filtered_data.append(hyperparameters)
            parameter_combinations_with_result_0.append(entry)

    # Create a plot to visualize parameter distribution
    parameter_names = list(filtered_data[0].keys())
    num_parameters = len(parameter_names)

    fig, axes = plt.subplots(nrows=1, ncols=num_parameters, figsize=(15, 5))
    for i, param_name in enumerate(parameter_names):
        param_values = [entry[param_name] for entry in filtered_data]
        axes[i].hist(param_values, bins=20)
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel("Frequency (result=0)")
        axes[i].set_title(f"{param_name} Distribution")

    plt.tight_layout()
    plt.show()

    # Summarize the data
    print("Parameter Distribution for Result=0:")
    for param_name in parameter_names:
        param_values = [entry[param_name] for entry in filtered_data]
        min_value = min(param_values)
        max_value = max(param_values)
        avg_value = sum(param_values) / len(param_values)
        print(f"{param_name}:")
        print(f"  Min: {min_value}")
        print(f"  Max: {max_value}")
        print(f"  Avg: {avg_value}\n")

    # Save all parameter combinations with Result=0 to another file
    with open(save_to, "w") as result_0_file:
        result_0_file.write("\n\n".join(parameter_combinations_with_result_0))

if __name__ == '__main__':
    N_P = 10
    N_Y = 5
    file_path = f"results/hyperparameter_results_UCB_NP{N_P}_NY{N_Y}.txt"
    save_to = f"results/parameter_combinations_with_result_UCB_NP{N_P}_NY{N_Y}.txt"
    result_analysis(file_path, save_to)