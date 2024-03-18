from utils.utils import *
# import sys
# sys.path.append('../utils/')  # Add the parent directory to the module search path

results = {
    "Test": "This is a test"
}

file_path = "test_scripts/test_results/"

file_name = "results.json"

previous_data = read_json_from_file(file_path + file_name)

print(previous_data)

experiment_id = len(previous_data['Experiments'])
results["Experiment_ID"] = experiment_id

previous_data['Experiments'].append(results)

data = previous_data

print(data)

if not os.path.exists(file_path):
    os.makedirs(file_path)

write_results_to_file(data, file_path + file_name)