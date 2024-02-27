_base_ = "/config/mmpretrain/swin_transformer/swin-base_16xb64_in1k.py"

load_from = "https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth"

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
        data_prefix="/data/UCMerced/UCMerced_LandUse/Images",
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

max_epochs = 200
train_cfg=dict(_delete_= True, max_epochs=max_epochs, type="EpochBasedTrainLoop")
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[160, 180],
        gamma=0.1
    )
]

work_dir = "./train_output"

test_dataloader = None
test_cfg = None
test_evaluator = None
val_dataloader = None
val_cfg = None
val_evaluator = None