from mmengine.config import Config
from mmengine.runner import Runner
import argparse
import os

DATA_NAME = os.environ.get("DATA_NAME", "CIFAR10")

def main():
    print(DATA_NAME)
    if DATA_NAME == "CIFAR10":
        # config = Config.fromfile('/rchristopher/data/src/mmpretrain_configs/convnext_cifar10_config.py')
        config = Config.fromfile('/rchristopher/data/src/mmpretrain_configs/convnext_cifar10_custom_config.py')
    elif DATA_NAME == "UCMerced_Landuse":
        config = Config.fromfile('/rchristopher/data/src/mmpretrain_configs/convnext_ucmerced_landuse_config.py')
        
    runner = Runner.from_cfg(config)
    runner.train()
    
if __name__ == "__main__":
    main()