import os

_base_ = "/rchristopher/data/src/mmpretrain/configs/convnext/convnext-base_32xb128_in21k.py"

load_from = "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_in21k_20220124-13b83eec.pth"

# Environment Variables
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 50))
DATA_NAME = os.environ.get("DATA_NAME", "UCMerced_Landuse")
MILESTONES = os.environ.get("MILESTONES",[10, 25, 50])
VISUALIZE = bool(os.environ.get("VISUALIZE", True))
OPTIMIZER = os.environ.get("OPTIMIZER", "SGD")
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.0001))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
FOLD_NUM = int(os.environ.get("FOLD_NUM", 0))

print("CONVNEXT UCMERCED LANDUSE CONFIG")

print(f"NUM_EPOCHS: {NUM_EPOCHS}")
print(f"DATA_NAME: {DATA_NAME}")
print(f"MILESTONES: {MILESTONES}")
print(f"VISUALIZE: {VISUALIZE}")
print(f"OPTIMIZER: {OPTIMIZER}")
print(f"LEARNING_RATE: {LEARNING_RATE}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"FOLD: {FOLD_NUM}")

data_type = "CustomDataset"
data_root = "/rchristopher/data/src/data/UCMerced_Landuse_5Folds/" + "fold_" + str(FOLD_NUM)
num_classes=21


data_preprocessor = dict(num_classes=num_classes)

train_pipeline = [
    {'type': 'LoadImageFromFile'},
    # {'type': 'RandomFlip', 'prob': 0.5, 'direction': 'horizontal'},
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
        # with_label=True,
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

# max_epochs = 200
train_cfg=dict(_delete_= True, max_epochs=NUM_EPOCHS, type="EpochBasedTrainLoop")
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=NUM_EPOCHS,
        by_epoch=True,
        milestones=MILESTONES,
        gamma=0.1
    )
]

visualization=dict(type='VisualizationHook', enable=VISUALIZE),

work_dir = "/rchristopher/data/src/mmpretrain_results/ucmerced_landuse/convnext/" + "fold_" + str(FOLD_NUM) + "/"

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    dataset=dict(
        type=data_type,
        data_root=data_root,
        data_prefix="test",
        # with_label=True,
        pipeline=train_pipeline
    )
)

val_evaluator = [
    dict(type='Accuracy', topk=(1, 5)),
    dict(type='ConfusionMatrix', num_classes=num_classes),
]

val_cfg = dict()

test_dataloader = val_dataloader
test_evaluator = val_evaluator
test_cfg = dict()