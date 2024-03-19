from mmengine.config import Config
from mmengine.runner import Runner
import argparse
import os

def main():
    current_directory = os.getcwd()
    print("Current Directory:", current_directory)

    # os.chdir('../mmpretrain_configs')
    # current_directory = os.getcwd()
    # print("Current Directory:", current_directory)

    files = os.listdir('../mmpretrain_configs')
    print("List of files in the directory:")
    for file in files:
        print(file)

    # config = Config.fromfile('../mmpretrain_configs/test_config.py')
    config = Config.fromfile('../mmpretrain_configs/ahurt_config.py')
    config.launcher = "pytorch"
    runner = Runner.from_cfg(config)
    runner.train()
    
if __name__ == "__main__":
    main()