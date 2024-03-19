import torch
import json
from matplotlib import pyplot as plt
import os

def cuda_setup() -> ():
    if torch.cuda.is_available():
        print(torch.cuda.current_device())     # The ID of the current GPU.
        print(torch.cuda.get_device_name(id))  # The name of the specified GPU, where id is an integer.
        print(torch.cuda.device(id))           # The memory address of the specified GPU, where id is an integer.
        print(torch.cuda.device_count())
        
    on_gpu = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    return device, on_gpu

def read_json_from_file(filename):
    # Check if the file exists
    if os.path.exists(filename):
        # If the file exists, read the existing JSON data
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
                if (data == None):
                    data = {"Experiments": []}
            except json.JSONDecodeError:
                # If the file is empty or contains invalid JSON, initialize data as an empty list
                data = {"Experiments": []}
    else:
        # If the file doesn't exist, initialize data as an empty list
        data = {"Experiments": []}

    return data

def write_results_to_file(results, filename):
    
    json_object = json.dumps(results, indent=4, default=lambda x: str(x))

    with open(filename, "w") as f:
        f.write(json_object)


def write_1data_plot_to_file(data_1, data_1_label, x_label, y_label, title, filename):
    fig = plt.figure()
    plt.plot(data_1, color='blue')
    plt.legend([data_1_label], loc='upper right')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.savefig(filename)


def write_2data_plot_to_file(data_1, data_1_label, data_2, data_2_label, x_label, y_label, title, filename):
    fig = plt.figure()
    plt.plot(data_1, color='blue')
    plt.plot(data_2, color='red')
    plt.legend([data_1_label, data_2_label], loc='upper right')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.savefig(filename)