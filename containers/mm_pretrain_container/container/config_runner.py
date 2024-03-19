from mmengine.config import Config
from mmengine.runner import Runner
import argparse
import os

def main():
    config = Config.fromfile('/data/convnext_config.py')
    runner = Runner.from_cfg(config)
    runner.train()
    
if __name__ == "__main__":
    main()