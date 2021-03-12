from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from rxnrep.scripts.launch_environment import PyTorchLaunch
from rxnrep.scripts.utils import load_checkpoint_wandb, save_files_to_wandb


def main(
    args,
    model,
    train_loader,
    val_loader,
    test_loader,
    top_k=3,
    monitor="val/score",
    project="tmp-rxnrep",
):

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor, mode="max", save_last=True, save_top_k=top_k, verbose=False
    )
    early_stop_callback = EarlyStopping(
        monitor=monitor, min_delta=0.0, patience=150, mode="max", verbose=True
    )

    # logger
    log_save_dir = Path("wandb").resolve()

    # restore model, epoch, shared_step, LR schedulers, apex, etc...
    if args.restore and log_save_dir.exists():
        # restore
        checkpoint_path, identifier = load_checkpoint_wandb(log_save_dir, project)
    else:
        # create new
        checkpoint_path = None
        identifier = None

    if not log_save_dir.exists():
        # put in try except in case it throws errors in distributed training
        try:
            log_save_dir.mkdir()
        except FileExistsError:
            pass
    wandb_logger = WandbLogger(save_dir=log_save_dir, project=project, id=identifier)

    # cluster environment to use torch.distributed.launch, e.g.
    # python -m torch.distributed.launch --use_env --nproc_per_node=2 <this_script.py>
    cluster = PyTorchLaunch()

    #
    # To run ddp on cpu, comment out `gpus` and `plugins`, and then set
    # `num_processes=2`, and `accelerator="ddp_cpu"`. Also note, for this script to
    # work, size of val (test) set should be larger than
    # `--num_centroids*num_processes`; otherwise clustering will raise an error,
    # but ddp_cpu cannot respond to it. As a result, it will stuck there.
    #

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        accelerator=args.accelerator,
        plugins=[cluster],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        resume_from_checkpoint=checkpoint_path,
        progress_bar_refresh_rate=100,
        flush_logs_every_n_steps=50,
        weights_summary="top",
        num_sanity_val_steps=0,  # 0, since we use centroids from training set
        # profiler="simple",
        # deterministic=True,
    )

    # ========== fit and test ==========
    trainer.fit(model, train_loader, val_loader)
    trainer.test(test_dataloaders=test_loader)

    # ========== save files to wandb ==========
    # Do not do this before trainer, since this might result in the initialization of
    # multiple wandb object when training in distribution mode
    if (
        args.gpus is None
        or args.gpus == 1
        or (args.gpus > 1 and cluster.local_rank() == 0)
    ):
        save_files_to_wandb(
            wandb_logger, [__file__, "running_metadata.yaml", "sweep.py", "submit.sh"]
        )
