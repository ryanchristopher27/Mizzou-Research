from nautiluslauncher import NautilusJobLauncher

namespace = "gp-engine-mu-hpdi-christopher"
job_prefix = "mm-convnext-ucmerced"
# command = ["ls"]
command = ["python3", "/rchristopher/data/src/code/mmpretrain_config_runner.py"]
image = "rchristopher27/rc-research-image:mmpretrain1"
pvc_name = "rc-large-pvc"

NUM_FOLDS = 5

defaults = dict(
    image=image,
    command=command,
    workingDir="/rchristopher/data",
    volumes={pvc_name: "/rchristopher/data"},
    shm=True,
    min_cpu=8,
    max_cpu=8,
    min_ram=8,
    max_ram=8,
    gpu=1,
    # gpu_types=["NVIDIA-A100-80GB-PCIe-MIG-1g.10gb"],
    env=dict(
        NUM_EPOCHS=100,
        DATA_NAME='UCMerced_Landuse',
        MILESTONES=[10, 25, 50],
        VISUALIZE=True,
        OPTIMIZER="SGD",
        LEARNING_RATE=0.0001,
        BATCH_SIZE=32,
    ),
)


jobs = [
    dict(job_name=job_prefix)
]

launcher = NautilusJobLauncher(
    cfg = dict(namespace=namespace, defaults=defaults, jobs=jobs)
)


# print(launcher.jobs)

launcher.run()