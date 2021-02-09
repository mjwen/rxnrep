import os
from datetime import datetime
from pathlib import Path

import psutil
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


def set_port(gpus, filename="current_port.txt"):
    """
    Set `MASTER_PORT`, using a value in `filename` and the input gpus.

    This is to deal with W&B sweep: if some sweep fail,

    Seems we can only do this because this is called before torch distributed groups
    are created, and thus we cannot init port on rank 1 and pass them to others.
    (Because dist.barrier cannot be used).

    The function relies on shared file, and check whether this is a handle on a file to
    avoid racing.

    Warnings:
        This only works for running on one node.

    Args:
        gpus: number of gpus, group of `gpus` values will have the same port
        filename: file to store the port into
    """

    path = Path(filename)

    while has_handle(str(path)):
        continue

    if not path.exists():
        fh = open(path, "w")
        port = 15000
        N = 0
    else:
        fh = open(path, "r+")
        line = fh.readlines()[-1].split()
        port = int(line[0])
        N = int(line[1])

    if N % gpus == 0:
        port += 1
    N += 1

    # set port
    os.environ["MASTER_PORT"] = str(port)

    # write current for next use
    fh.write(f"{port} {N}  {datetime.now()}\n")

    fh.close()


def has_handle(fpath):
    """
    Whether there is a handle on a file.
    """
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if fpath == item.path:
                    return True
        except Exception:
            pass

    return False


if __name__ == "__main__":
    for i in range(20):
        set_port(2)
