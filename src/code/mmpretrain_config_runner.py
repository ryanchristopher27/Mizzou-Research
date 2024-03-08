from mmengine.config import Config
from mmengine.runner import Runner
import argparse
import os

def main():
    config = Config.fromfile('/data/src/mmpretrain_configs/convnext_ucmercedlanduse_config.py')
    runner = Runner.from_cfg(config)
    runner.train()
    
if __name__ == "__main__":
    main()