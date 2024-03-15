import json

from mmengine.config import Config
cfg = Config.fromfile("../../mmpretrain/configs/convnext/convnext-base_32xb128_in21k.py")
# cfg = Config.fromfile("../mmpretrain_configs/ahurt_config.py")

print(json.dumps(dict(cfg), indent=2))

cfg['work_dir'] = './data'
cfg['val_cfg'] = None
cfg['test_cfg'] = None
cfg['visualizer'] = None

from mmengine.runner import Runner
runner = Runner.from_cfg(cfg)
runner.train()