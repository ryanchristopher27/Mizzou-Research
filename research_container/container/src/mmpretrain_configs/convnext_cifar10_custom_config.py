import os

_base_ = "/rchristopher/data/src/mmpretrain/configs/convnext/convnext-base_32xb128_in21k.py"

load_from = "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_in21k_20220124-13b83eec.pth"

# Environment Variables
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 50))
DATA_NAME = os.environ.get("DATA_NAME", "CIFAR10")
MILESTONES = os.environ.get("MILESTONES",[10, 25, 50])
VISUALIZE = bool(os.environ.get("VISUALIZE", True))
OPTIMIZER = os.environ.get("OPTIMIZER", "SGD")
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.0001))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))

print("CONVNEXT CIFAR10 CONFIG")

print(f"NUM_EPOCHS: {NUM_EPOCHS}")
print(f"DATA_NAME: {DATA_NAME}")
print(f"MILESTONES: {MILESTONES}")
print(f"VISUALIZE: {VISUALIZE}")
print(f"OPTIMIZER: {OPTIMIZER}")
print(f"LEARNING_RATE: {LEARNING_RATE}")
print(f"BATCH_SIZE: {BATCH_SIZE}")

data_type = "CustomDataset"
data_root = "/rchristopher/data/src/data/CIFAR10"
num_classes=10

data_preprocessor = dict(num_classes=num_classes)
# data_preprocessor = dict(
#     num_classes=num_classes,
#     mean=[0.485, 0.456, 0.406], 
#     std=[0.229, 0.224, 0.225], 
#     to_rgb=True,
# )

# train_pipeline = [
#     {'type': 'LoadImageFromFile'},
#     # {'type': 'RandomFlip', 'prob': 0.5, 'direction': 'horizontal'},
#     {'type': 'PackInputs'}
# ]
train_pipeline = [
    {'type': 'LoadImageFromFile'},
    {'type': 'Resize', 'scale': 224},
    {'type': 'CenterCrop', 'crop_size': 224},
    {'type': 'Normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    {'type': 'PackInputs'}
]

model = dict(
    head=dict(
        num_classes=num_classes,
    )
)

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    dataset=dict(
        _delete_=True,
        type=data_type,
        data_root=data_root,
        data_prefix="train",
        with_label=True,
        pipeline=train_pipeline
    )
)

optim_wrapper=dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type=OPTIMIZER,
        lr=LEARNING_RATE,
        weight_decay=0.0001
    ),
)

train_cfg=dict(_delete_= True, max_epochs=NUM_EPOCHS, type="EpochBasedTrainLoop", val_interval=1)

# param_scheduler = [
#     dict(
#         type="MultiStepLR",
#         begin=0,
#         end=NUM_EPOCHS,
#         by_epoch=True,
#         milestones=MILESTONES,
#         gamma=0.1
#     )
# ]

# ====================================================
# New Param Scheduler from Keli
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=NUM_EPOCHS,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=20)
]
# ====================================================

# default_hooks = dict(
#     checkpoint=dict(type='CheckpointHook', interval=-1),
#     visualization=dict(type='VisualizationHook', enable=VISUALIZE),
# )

# ====================================================
# Default Hooks from Keli
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)
# ====================================================


work_dir = "/rchristopher/data/src/mmpretrain_results/cifar10/convnext/"
# work_dir = "/rchristopher/data/src/results/mmpretrain_results"


val_dataloader = dict(
    batch_size=BATCH_SIZE,
    dataset=dict(
        type=data_type,
        data_root=data_root,
        data_prefix="test",
        with_label=True,
        pipeline=train_pipeline
    )
)

val_evaluator = [
    dict(type='Accuracy', topk=(1)),
    dict(type='ConfusionMatrix', num_classes=num_classes),
]

# val_evaluator = dict(type='ConfusionMatrix', num_classes=num_classes)

val_cfg = dict()

test_dataloader = val_dataloader
test_evaluator = val_evaluator
test_cfg = dict()

'''
val_dataloader = None
val_evaluator = None
val_cfg = None
test_dataloader = None
test_evaluator = None
test_cfg = None
'''