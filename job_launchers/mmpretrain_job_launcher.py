from nautiluslauncher import NautilusJobLauncher

namespace = "gp-engine-mu-hpdi-christopher"
job_prefix = "mmpretrain-job"
command = ["python3", "/rchristopher/data/src/code/mmpretrain_config_runner.py"]
image = "rchristopher27/rc-research-image:mmpretrain2"
pvc_name = "rc-large-pvc"

NUM_FOLDS = 5
DATA_NAME = 'UCMerced_Landuse'

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
        NUM_EPOCHS=100,
        DATA_NAME=DATA_NAME,
        MILESTONES=[10, 25, 50],
        VISUALIZE=True,
        OPTIMIZER="SGD",
        LEARNING_RATE=0.001,
        BATCH_SIZE=16,
        PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128",
    ),
)

if DATA_NAME == 'CIFAR10':
    jobs = [
        dict(job_name=job_prefix)
    ]
elif DATA_NAME == 'UCMerced_Landuse':
    # All Folds
    jobs = [
        dict(job_name=job_prefix + "-" + str(i), env=dict(
            FOLD_NUM=i,
        ))
        for i in range(NUM_FOLDS)
    ]

    # Single Fold
    # FOLD = 2
    # jobs = [
    #     dict(job_name=job_prefix + "-" + str(FOLD), env=dict(
    #         FOLD_NUM=FOLD,
    #     ))
    # ]

launcher = NautilusJobLauncher(
    cfg = dict(namespace=namespace, defaults=defaults, jobs=jobs)
)


# print(launcher.jobs)

launcher.run()