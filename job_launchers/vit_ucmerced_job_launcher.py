from nautiluslauncher import NautilusJobLauncher

namespace = "gp-engine-mu-hpdi-christopher"
job_prefix = "vit-ucmerced-"
command = ["python3", "/data/Vit_Kfold_UCMerced.py"]
image = "rchristopher27/cnn-image:latest"
pvc_name = "rchristopher-pvc"

NUM_FOLDS = 5

defaults = dict(
    image=image,
    command=command,
    workingDir="/data",
    volumes={pvc_name: "/data"},
    shm=True,
    min_cpu=8,
    max_cpu=8,
    min_ram=8,
    max_ram=8,
    gpu=1,
    # gpu_types=["NVIDIA-A100-80GB-PCIe-MIG-1g.10gb"],
    env=dict(
        TORCH_NUM_JOBS=8, 
        TORCH_NUM_EPOCHS=50,
        TORCH_NUM_FOLDS=NUM_FOLDS,
        WRITE_RESULTS=True,
        ),
)

jobs = [
    dict(job_name=job_prefix + str(i+1), env=dict(
        FOLD_NUM=i+1,
        TORCH_MODEL_NAME="resnet50",
        TORCH_DATA_NAME="ucmerced_landuse",
    ))
    for i in range(NUM_FOLDS)
]

# Single Job Test
# jobs = [
#     dict(job_name='test-job', env=dict(
#         FOLD_NUM=1,
#         TORCH_MODEL_NAME="resnet50",
#         TORCH_DATA_NAME="ucmerced_landuse",
#     ))
# ]

launcher = NautilusJobLauncher(
    cfg = dict(namespace=namespace, defaults=defaults, jobs=jobs)
)


# print(launcher.jobs)

launcher.run()