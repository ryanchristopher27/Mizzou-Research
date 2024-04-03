from nautiluslauncher import NautilusJobLauncher
from kubernetes.client import V1Affinity, V1NodeAffinity, V1NodeSelector, V1NodeSelectorTerm, V1NodeSelectorRequirement


namespace = "gp-engine-mu-hpdi-christopher"
job_prefix = "mmpretrain-job"
command = ["python3", "/rchristopher/data/src/code/mmpretrain_config_runner.py"]
image = "rchristopher27/rc-research-image:mmpretrain2"
pvc_name = "rc-large-pvc"

NUM_FOLDS = 5
# DATA_NAME = 'UCMerced_Landuse'
DATA_NAME = 'CIFAR10'

gpu_types = [
    "NVIDIA-A100-SXM4-80GB",
    "NVIDIA-A40",
    "NVIDIA-A100-80GB-PCIe",
    "NVIDIA-RTX-A4000"
]

affinity = V1Affinity(
    node_affinity=V1NodeAffinity(
        required_during_scheduling_ignored_during_execution=V1NodeSelector(
            node_selector_terms=[
                V1NodeSelectorTerm(
                    match_expressions=[
                        V1NodeSelectorRequirement(
                            key="nvidia.com/gpu.product",
                            operator="In",
                            values=gpu_types,
                        ),
                        # V1NodeSelectorRequirement(
                        #     key="topology.kubernetes.io/zone",
                        #     operator="NotIn",
                        #     values=["ucsd-nrp"]
                        # ),
                        V1NodeSelectorRequirement(
                            key="topology.kubernetes.io/region",
                            operator="In",
                            values=["us-central", "us-west"]
                        )
                    ]
                )
            ]
        )
    )
)

defaults = dict(
    image=image,
    command=command,
    workingDir="/rchristopher/data",
    volumes={pvc_name: "/rchristopher/data"},
    shm=True,
    min_cpu=2,
    max_cpu=4,
    # min_ram=24,
    # max_ram=36,
    min_ram=12,
    max_ram=18,
    gpu=1,
    gpu_types=gpu_types,
    # gpu_types=["NVIDIA-A100-80GB-PCIe-MIG-1g.10gb"],
    env=dict(
        NUM_EPOCHS=100,
        DATA_NAME=DATA_NAME,
        MILESTONES=[10, 25, 50],
        VISUALIZE=True,
        OPTIMIZER="SGD",
        LEARNING_RATE=0.001,
        BATCH_SIZE=64,
        PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128",
    ),
)

# if DATA_NAME == 'CIFAR10':
#     jobs = [
#         # dict(job_name=job_prefix + "-c10-512", affinity=affinity)
#         dict(job_name=job_prefix + "-c10", affinity=affinity)
#     ]
# elif DATA_NAME == 'UCMerced_Landuse':
#     # All Folds
#     # jobs = [
#     #     dict(job_name=job_prefix + "-um-" + str(i), env=dict(
#     #         FOLD_NUM=i,
#     #     ))
#     #     for i in range(NUM_FOLDS)
#     # ]

#     # Single Fold
#     FOLD = 2
#     jobs = [
#         dict(job_name=job_prefix + "-um-" + str(FOLD), env=dict(
#             FOLD_NUM=FOLD,
#         ), affinity=affinity)
#     ]

def update_job_spec(job_spec, affinity):
    if job_spec.get('gpu', 0) > 0 and 'gpu_types' in job_spec:
        job_spec['affinity'] = affinity
    return job_spec

if DATA_NAME == 'CIFAR10':
    jobs = [
        update_job_spec(dict(job_name=job_prefix + "-c10"), affinity)
    ]
elif DATA_NAME == 'UCMerced_Landuse':
    FOLD = 2
    jobs = [
        update_job_spec(dict(job_name=job_prefix + "-um-" + str(FOLD), env=dict(
            FOLD_NUM=FOLD,
        )), affinity)
    ]

launcher = NautilusJobLauncher(
    cfg = dict(namespace=namespace, defaults=defaults, jobs=jobs)
)


# print(launcher.jobs)

launcher.run()