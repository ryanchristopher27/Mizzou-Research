from nautiluslauncher import NautilusJobLauncher

namespace = "gp-engine-mu-hpdi-christopher"
job_prefix = "ft-"
command = ["python3", "/rchristopher/data/src/fine_tuning_experiment/ft_code/fine_tuning_experiment.py"]
image = "rchristopher27/rc-research-image:finetuning1"
pvc_name = "rc-large-pvc"

NUM_FOLDS = 5

dataset = "ucmerced_landuse"

defaults = dict(
    image=image,
    command=command,
    workingDir="/rchristopher/data",
    volumes={pvc_name: "/rchristopher/data"},
    shm=True,
    min_cpu=4,
    max_cpu=10,
    min_ram=8,
    max_ram=10,
    gpu=1,
    # gpu_types=["NVIDIA-A100-80GB-PCIe-MIG-1g.10gb"],
    env=dict(
        TORCH_NUM_JOBS=8, 
        TORCH_NUM_EPOCHS=100,
        TORCH_NUM_FOLDS=NUM_FOLDS,
        WRITE_RESULTS=True,
        TORCH_MODEL_NAME="vit_b_32",
        TORCH_DATA_NAME=dataset,
        # OPTIMIZER="SGD",
        LOSS_FUNCTION="CrossEntropy",
        # BATCH_SIZE=16,
        # LEARNING_RATE=0.001,
        PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128",
        ),
)


# =====================================================================
# Multiple Jobs Based on Different Hyperparameters
# =====================================================================
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
optimizers = ["SGD", "Adam", "AdamW"]
batch_sizes = [16, 32]

run_all = True

jobs = []
job_counter = 1
for lr in learning_rates:
    for optimizer in optimizers:
        for batch_size in batch_sizes:
            if dataset == "ucmerced_landuse":
                for fold in range(NUM_FOLDS):
                    if job_counter in [104, 108, 120, 123, 40, 87, 91] or run_all: # Used for single run
                        temp_dict = dict(job_name=job_prefix + str(job_counter) + "-" + str(fold+1), env=dict(
                            FOLD_NUM=fold+1,
                            OPTIMIZER=optimizer,
                            LEARNING_RATE=lr,
                            BATCH_SIZE=batch_size,
                            EXPERIMENT=job_counter,
                        ))
                        jobs.append(temp_dict)


                    job_counter += 1

            elif dataset == "cifar10":
                if job_counter in [] or run_all:
                    temp_dict = dict(job_name=job_prefix + str(job_counter), env=dict(
                        FOLD_NUM=1,
                        OPTIMIZER=optimizer,
                        LEARNING_RATE=lr,
                        BATCH_SIZE=batch_size,
                        EXPERIMENT=job_counter,
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