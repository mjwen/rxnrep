import os
from pytorch_lightning.cluster_environments import ClusterEnvironment


class PyTorchLaunch(ClusterEnvironment):
    """
    A cluster environment to use the environment variables set by:
    `torch.distributed.launch`.

    Then, we can submit lightning script on slurm using something like:

    python -m torch.distributed.launch  --use_env --nproc_per_node=NUM_GPUS_YOU_HAVE
           --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"

    instead of using `srun`.

    In this way, the process for each GPU is created by torch.distributed.launch
    instead of by `srun`.

    The main purpose is to use `num_workers=0` in dataloader for `small` dataset that
    can be placed in memory. (We noticed that using multiple workers for dataloading
    increases the memory, which is unnecessary for dataset that is already in memory.)
    """

    def master_address(self):
        return os.environ["MASTER_ADDR"]

    def master_port(self):
        return int(os.environ["MASTER_PORT"])

    def world_size(self):
        return int(os.environ["WORLD_SIZE"])

    def local_rank(self):
        return int(os.environ["LOCAL_RANK"])
