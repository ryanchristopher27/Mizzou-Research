from nautiluslauncher import NautilusJobLauncher

namespace = "gp-engine-mu-hpdi-christopher"
job_prefix = "kfold-"
command = ["python3", "/data/Assorted_Kfold_Analysis.py"]
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
        TORCH_NUM_EPOCHS=40,
        TORCH_NUM_FOLDS=NUM_FOLDS,
        WRITE_RESULTS=True,
        OPTIMIZER="Adam",
        LOSS_FUNCTION="CrossEntropy",
        BATCH_SIZE=128,
        LEARNING_RATE=0.001,
        ),
)

models = ["vit_b_16", "vit_b_32"]
datasets = ["ucmerced_landuse", "cifar10"]

jobs = []
job_counter = 1
for model in models:
    for dataset in datasets:
        for fold in range(NUM_FOLDS):
            temp_dict = dict(job_name=job_prefix + str(job_counter), env=dict(
                FOLD_NUM=fold+1,
                TORCH_MODEL_NAME=model,
                TORCH_DATA_NAME=dataset,
            ))

            jobs.append(temp_dict)

            job_counter += 1

# jobs = [
#     dict(job_name=job_prefix + str(i+1), env=dict(
#         FOLD_NUM=i+1,
#         TORCH_MODEL_NAME="resnet50",
#         TORCH_DATA_NAME="ucmerced_landuse",
#     ))
#     for i in range(NUM_FOLDS)
# ]

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