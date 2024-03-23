from nautiluslauncher import NautilusJobLauncher

namespace = "gp-engine-mu-hpdi-christopher"
job_prefix = "dl-"
command = ["python3", "/rchristopher/data/src/code/Assorted_DL_Analysis.py"]
image = "rchristopher27/rc-research-image:generic1"
pvc_name = "rc-large-pvc"

NUM_FOLDS = 5

defaults = dict(
    image=image,
    command=command,
    workingDir="/rchristopher/data",
    volumes={pvc_name: "/rchristopher/data"},
    shm=True,
    min_cpu=2,
    max_cpu=4,
    min_ram=12,
    max_ram=18,
    gpu=1,
    # gpu_types=["NVIDIA-A100-80GB-PCIe-MIG-1g.10gb"],
    env=dict(
        TORCH_NUM_JOBS=8, 
        TORCH_NUM_EPOCHS=100,
        TORCH_NUM_FOLDS=NUM_FOLDS,
        WRITE_RESULTS=True,
        OPTIMIZER="SGD",
        LOSS_FUNCTION="CrossEntropy",
        BATCH_SIZE=32,
        LEARNING_RATE=0.001,
        PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128",
        ),
)


# =====================================================================
# Multiple Jobs Based on Above Models/Datasets
# =====================================================================
models = ["resnet18", "resnet50", "vit_b_16", "vit_b_32"]
datasets = ["ucmerced_landuse"] # ['cifar10', 'ucmerced_landuse']

jobs = []
job_counter = 1
for model in models:
    for dataset in datasets:
        if dataset == "cifar10":
            temp_dict = dict(job_name=job_prefix + str(job_counter), env=dict(
                FOLD_NUM=1,
                TORCH_MODEL_NAME=model,
                TORCH_DATA_NAME=dataset,
            ))

            jobs.append(temp_dict)

            job_counter += 1
        
        elif dataset == "ucmerced_landuse":
            for fold in range(NUM_FOLDS):
                temp_dict = dict(job_name=job_prefix + model.replace("_", "-") + "-" + str(fold+1), env=dict(
                    FOLD_NUM=fold+1,
                    TORCH_MODEL_NAME=model,
                    TORCH_DATA_NAME=dataset,
                ))

                jobs.append(temp_dict)

                job_counter += 1

# =====================================================================
# K-Fold Jobs
# =====================================================================
# jobs = [
#     dict(job_name=job_prefix + str(i+1), env=dict(
#         FOLD_NUM=i+1,
#         TORCH_MODEL_NAME="vit_b_16",
#         TORCH_DATA_NAME="cifar10",
#     ))
#     for i in range(NUM_FOLDS)
# ]

# =====================================================================
# Single Job Test
# =====================================================================
# jobs = [
#     dict(job_name='single-job', env=dict(
#         FOLD_NUM=1,
#         TORCH_MODEL_NAME="vit_b_16",
#         TORCH_DATA_NAME="ucmerced_landuse",
#     ))
# ]

launcher = NautilusJobLauncher(
    cfg = dict(namespace=namespace, defaults=defaults, jobs=jobs)
)


# print(launcher.jobs)

launcher.run()