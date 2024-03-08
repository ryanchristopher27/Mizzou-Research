from nautiluslauncher import NautilusJobLauncher

namespace = "gp-engine-mu-hpdi-christopher"
job_prefix = "mmpretrain-convnext-ucmerced"
# command = ["ls"]
command = ["python3", "/data/src/code/mmpretrain_config_runner.py"]
image = "gitlab-registry.nrp-nautilus.io/jhurt/mmdet-v3/mmdet-base:v3.1"
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
    env=dict(),
)


jobs = [
    dict(job_name=job_prefix)
]

launcher = NautilusJobLauncher(
    cfg = dict(namespace=namespace, defaults=defaults, jobs=jobs)
)


# print(launcher.jobs)

launcher.run()