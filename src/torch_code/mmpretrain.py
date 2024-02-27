from mmengine.config import Config
cfg = Config.fromfile("../MMPretrain/configs/convnext-ucmercedlanduse.py")
from mmengine.runner import Runner
runner = Runner.from_cfg(cfg)
runner.train()