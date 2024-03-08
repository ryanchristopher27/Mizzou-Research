_base_ = "/data/src/mmpretrain/configs/convnext/convnext-base_32xb128_in21k.py"

load_from = "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_in21k_20220124-13b83eec.pth"

data_preprocessor = dict(num_classes=21)

train_pipeline = [
    {'type': 'LoadImageFromFile'},
    {'type': 'RandomFlip', 'prob': 0.5, 'direction': 'horizontal'},
    {'type': 'PackInputs'}
]

model = dict(
    head=dict(
        num_classes=21,
    )
)

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        _delete_=True,
        type="CustomDataset",
        data_prefix="/data/src/data/UCMerced_Landuse/",
        with_label=True,
        pipeline=train_pipeline
    )
)


optim_wrapper=dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='SGD',
        lr=0.0001,
        weight_decay=0.0001
    ),
)

max_epochs = 2
train_cfg=dict(_delete_= True, max_epochs=max_epochs, type="EpochBasedTrainLoop")
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[1],
        gamma=0.1
    )
]

work_dir = "/data/src/results/mmpretrain_results"

test_dataloader = None
test_cfg = None
test_evaluator = None
val_dataloader = None
val_cfg = None
val_evaluator = None