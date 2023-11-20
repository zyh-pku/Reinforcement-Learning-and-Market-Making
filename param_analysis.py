import matplotlib.pyplot as plt
import re

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

# Read the contents of the text file
with open("hyperparameter_results.txt", "r") as file:
    lines = file.read()

# Split the text into individual result entries
result_entries = re.split(r'\n(?=Hyperparameters:)', lines)

# Initialize lists to store filtered data and all parameter combinations with Result=0
filtered_data = []
parameter_combinations_with_result_0 = []

# Parse the data and filter based on Result
for entry in result_entries:
    hyperparameters, result, runtime = extract_hyperparameters_and_results(entry)
    if result == 0:
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
with open("parameter_combinations_with_result_0.txt", "w") as result_0_file:
    result_0_file.write("\n\n".join(parameter_combinations_with_result_0))
