import torch

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