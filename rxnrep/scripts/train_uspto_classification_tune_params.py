import sys
import warnings
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import optuna
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from rxnrep.data.uspto import SchneiderDataset
from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.model.model import LinearClassification
from rxnrep.model.metric import MultiClassificationMetrics
from rxnrep.scripts.utils import (
    EarlyStopping,
    seed_all,
    save_checkpoints,
    load_checkpoints,
    load_part_pretrained_model,
)

from rxnrep.scripts.utils import init_distributed_mode, ProgressMeter, TimeMeter
from rxnrep.utils import yaml_dump, to_path


def parse_args():
    parser = argparse.ArgumentParser(description="Reaction Representation")

    # ========== input files ==========
    prefix = "/Users/mjwen/Documents/Dataset/uspto/Schneider50k/"

    fname_tr = prefix + "schneider_n200_processed_train_label_manipulated.tsv"
    fname_val = prefix + "schneider_n200_processed_val.tsv"
    fname_test = prefix + "schneider_n200_processed_test.tsv"

    parser.add_argument("--trainset-filename", type=str, default=fname_tr)
    parser.add_argument("--valset-filename", type=str, default=fname_val)
    parser.add_argument("--testset-filename", type=str, default=fname_test)

    # ========== embedding layer ==========
    parser.add_argument("--embedding-size", type=int, default=24)

    # ========== encoder ==========
    parser.add_argument(
        "--molecule-conv-layer-sizes", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--molecule-num-fc-layers", type=int, default=2)
    parser.add_argument("--molecule-batch-norm", type=int, default=1)
    parser.add_argument("--molecule-activation", type=str, default="ReLU")
    parser.add_argument("--molecule-residual", type=int, default=1)
    parser.add_argument("--molecule-dropout", type=float, default="0.0")
    parser.add_argument(
        "--reaction-conv-layer-sizes", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--reaction-num-fc-layers", type=int, default=2)
    parser.add_argument("--reaction-batch-norm", type=int, default=1)
    parser.add_argument("--reaction-activation", type=str, default="ReLU")
    parser.add_argument("--reaction-residual", type=int, default=1)
    parser.add_argument("--reaction-dropout", type=float, default="0.0")

    # linear classification head
    parser.add_argument(
        "--head-hidden-layer-sizes", type=int, nargs="+", default=[256, 128]
    )
    parser.add_argument("--head-activation", type=str, default="ReLU")
    parser.add_argument("--num-classes", type=int, default=50)
    parser.add_argument(
        "--only-train-classification-head",
        type=int,
        default=0,
        help="whether to only train the classification head",
    )

    # ========== training ==========
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--pretrained-model-filename",
        type=str,
        default=None,
        help="Path to the checkpoint of the pretrained model to use part of its "
        "parameters. If `None`, will not use it.",
    )

    parser.add_argument("--restore", type=int, default=0, help="restore training")
    parser.add_argument(
        "--dataset-state-dict-filename", type=str, default="dataset_state_dict.yaml"
    )

    # ========== distributed ==========
    parser.add_argument(
        "--launch-mode",
        type=str,
        default="spawn",
        help="How to launch distributed training: [`torch_launch`| `srun` | `spawn`]",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Whether to use GPU.")
    parser.add_argument(
        "--distributed", type=int, default=0, help="Whether distributed DDP training."
    )
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument(
        "--dist-url",
        type=str,
        help="tcp port for distributed communication e.g. `tcp://localhost:15678`.",
    )
    parser.add_argument("--local_rank", type=int, help="Local rank of process.")

    args = parser.parse_args()

    return args


def load_dataset(args):

    # check dataset state dict if restore model
    if args.restore:
        if args.dataset_state_dict_filename is None:
            warnings.warn(
                "Restore with `args.dataset_state_dict_filename` set to None."
            )
            state_dict_filename = None
        elif not Path(args.dataset_state_dict_filename).exists():
            warnings.warn(
                f"args.dataset_state_dict_filename: `{args.dataset_state_dict_filename} "
                "not found; set to `None`."
            )
            state_dict_filename = None
        else:
            state_dict_filename = args.dataset_state_dict_filename
    else:
        state_dict_filename = None

    trainset = SchneiderDataset(
        filename=args.trainset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=state_dict_filename,
    )
    valset = SchneiderDataset(
        filename=args.valset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=trainset.state_dict(),
    )
    testset = SchneiderDataset(
        filename=args.testset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=trainset.state_dict(),
    )

    # save dataset state dict for retraining or prediction
    if not args.distributed or args.rank == 0:
        trainset.save_state_dict_file(args.dataset_state_dict_filename)
        print(
            "Trainset size: {}, valset size: {}: testset size: {}.".format(
                len(trainset), len(valset), len(testset)
            )
        )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=trainset.collate_fn,
        drop_last=False,
    )

    # TODO, for val set, we can also make it distributed and report the error on rank
    #  0. If the vl set size is large enough, the statistics of error should be the same
    #  in different ranks. If this is not good, we can gather and reduce the validation
    #  metric.

    # larger val and test set batch_size is faster but needs more memory
    # adjust the batch size of to fit memory
    bs = max(len(valset) // 10, 1)
    val_loader = DataLoader(
        valset,
        batch_size=bs,
        shuffle=False,
        collate_fn=valset.collate_fn,
        drop_last=False,
    )
    bs = max(len(testset) // 10, 1)
    test_loader = DataLoader(
        testset,
        batch_size=bs,
        shuffle=False,
        collate_fn=testset.collate_fn,
        drop_last=False,
    )

    # set args for model
    args.feature_size = trainset.feature_size

    return train_loader, val_loader, test_loader, train_sampler


def train(optimizer, model, data_loader, class_weight, epoch, args):
    timer = TimeMeter(frequency=5)

    model.train()

    nodes = ["atom", "bond", "global"]

    # class weights
    class_weight = class_weight.to(args.device)

    if not args.distributed or args.rank == 0:
        timer.display(epoch, f"In epoch; class weight")

    # evaluation metrics
    metrics = MultiClassificationMetrics(num_classes=args.num_classes)

    if not args.distributed or args.rank == 0:
        timer.display(epoch, f"In epoch; clustering")

    epoch_loss = 0.0
    for it, (indices, mol_graphs, rxn_graphs, labels, metadata) in enumerate(
        data_loader
    ):

        if not args.distributed or args.rank == 0:
            timer.display(it, f"Batch {it}; getting data")

        mol_graphs = mol_graphs.to(args.device)
        rxn_graphs = rxn_graphs.to(args.device)
        feats = {
            nt: mol_graphs.nodes[nt].data.pop("feat").to(args.device) for nt in nodes
        }
        labels = labels["reaction_class"].to(args.device)

        if not args.distributed or args.rank == 0:
            timer.display(it, f"Batch {it}; to cuda")

        preds = model(mol_graphs, rxn_graphs, feats, metadata)

        if not args.distributed or args.rank == 0:
            timer.display(it, f"Batch {it}; model predict")

        loss = F.cross_entropy(preds, labels, reduction="mean", weight=class_weight)

        if not args.distributed or args.rank == 0:
            timer.display(it, f"Batch {it}; computing loss")

        # ========== update model parameters ==========
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

        if not args.distributed or args.rank == 0:
            timer.display(it, f"Batch {it}; back propagation")

        # ========== metrics ==========
        p = torch.argmax(preds, dim=1)
        metrics.step(p, labels)

        if not args.distributed or args.rank == 0:
            timer.display(it, f"Batch {it}; keep data for metric")

    if not args.distributed or args.rank == 0:
        timer.display(epoch, f"In epoch; Finish looping batch")

    # compute metric values
    epoch_loss /= it + 1
    metrics.compute_metric_values(class_reduction="weighted")

    if not args.distributed or args.rank == 0:
        timer.display(epoch, f"In epoch; Compute metric")

    return epoch_loss, metrics


def evaluate(model, data_loader, args):
    model.eval()

    nodes = ["atom", "bond", "global"]

    # evaluation metrics
    metrics = MultiClassificationMetrics(num_classes=args.num_classes)

    with torch.no_grad():

        for it, (indices, mol_graphs, rxn_graphs, labels, metadata) in enumerate(
            data_loader
        ):
            mol_graphs = mol_graphs.to(args.device)
            rxn_graphs = rxn_graphs.to(args.device)
            feats = {
                nt: mol_graphs.nodes[nt].data.pop("feat").to(args.device)
                for nt in nodes
            }
            labels = labels["reaction_class"].to(args.device)

            preds = model(mol_graphs, rxn_graphs, feats, metadata)

            # ========== metrics ==========
            # bond type
            p = torch.argmax(preds, dim=1)
            metrics.step(p, labels)

    # compute metric values
    metrics.compute_metric_values(class_reduction="weighted")

    return metrics


def main(rank, trial, args):
    args.rank = rank

    if args.distributed:
        port = 15678 + trial.number
        args.dist_url = f"tcp://localhost:{port}"
        init_distributed_mode(args)
    else:
        if args.gpu:
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cpu")

    if not args.distributed or args.rank == 0:
        print(args)
        print("\n\nStart training at:", datetime.now())

    # Explicitly setting seed to ensure the same dataset split and models created in
    # two processes (when distributed) starting from the same random weights and biases
    seed_all()

    ################################################################################
    #  set up dataset, model, optimizer ...
    ################################################################################

    # dataset
    train_loader, val_loader, test_loader, train_sampler = load_dataset(args)

    # save args (need to do this here since additional args are attached in load_dataset)
    if not args.distributed or args.rank == 0:
        yaml_dump(args, "train_args.yaml")

    # model
    model = LinearClassification(
        in_feats=args.feature_size,
        embedding_size=args.embedding_size,
        # encoder
        molecule_conv_layer_sizes=args.molecule_conv_layer_sizes,
        molecule_num_fc_layers=args.molecule_num_fc_layers,
        molecule_batch_norm=args.molecule_batch_norm,
        molecule_activation=args.molecule_activation,
        molecule_residual=args.molecule_residual,
        molecule_dropout=args.molecule_dropout,
        reaction_conv_layer_sizes=args.reaction_conv_layer_sizes,
        reaction_num_fc_layers=args.reaction_num_fc_layers,
        reaction_batch_norm=args.reaction_batch_norm,
        reaction_activation=args.reaction_activation,
        reaction_residual=args.reaction_residual,
        reaction_dropout=args.reaction_dropout,
        # classification head
        head_hidden_layer_sizes=args.head_hidden_layer_sizes,
        num_classes=args.num_classes,
        head_activation=args.head_activation,
    )

    if not args.distributed or args.rank == 0:
        print(model)

    # freeze parameters (this should come before DDP, since attributes like encoder
    # cannot be directly accessed in DDP model)
    if args.only_train_classification_head:
        # freeze encoder parameters
        for p in model.encoder.parameters():
            p.requires_grad = False
        # freeze set2set parameters
        for p in model.set2set.parameters():
            p.requires_grad = False

        # check only classification head params is trained
        num_params_classification_head = sum(
            p.numel() for p in model.classification_head.parameters()
        )
        num_params_trainable = sum(
            [p.numel() for p in model.parameters() if p.requires_grad]
        )
        assert (
            num_params_classification_head == num_params_trainable
        ), "parameters other than classification head are trained"

        if not args.distributed or args.rank == 0:
            print("\nFreeze some parameters to only train classification head...")

    # load pretrained models
    # (this is typically used together with `only_train_classification_head`)
    if args.pretrained_model_filename is not None:
        load_part_pretrained_model(
            model, map_location=args.device, filename=args.pretrained_model_filename
        )
        if not args.distributed or args.rank == 0:
            print("\nLoad pretrained model...")

    # DDP
    if args.distributed:
        if args.gpu:
            model.to(args.device)
            ddp_model = DDP(model, device_ids=[args.device])
        else:
            ddp_model = DDP(model)
        model = ddp_model
    else:
        if args.gpu:
            model.to(args.device)

    # optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # learning rate scheduler and stopper
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.4, patience=50, verbose=True
    )
    stopper = EarlyStopping(patience=150)

    # class weight
    class_weight = train_loader.dataset.get_class_weight(num_classes=args.num_classes)

    best = -np.finfo(np.float32).max

    # load checkpoint
    # Note this is overwrite the parameters loaded when `args.pretrained_model_filename`
    # is not None. Most of times, this is fine, since when we specify `restore`,
    # the parameters in the pretrained model is typically already loaded in the
    # previous run.
    state_dict_objs = {"model": model, "optimizer": optimizer, "scheduler": scheduler}
    if args.restore:
        try:
            checkpoint = load_checkpoints(
                state_dict_objs, map_location=args.device, filename="checkpoint.pkl"
            )

            args.start_epoch = checkpoint["epoch"]
            best = checkpoint["best"]
            print(
                f"Successfully load checkpoints, best {best}, epoch {args.start_epoch}"
            )

        except FileNotFoundError as e:
            warnings.warn(str(e) + " Continue without loading checkpoints.")
            pass

    ################################################################################
    # training loop
    ################################################################################
    progress = ProgressMeter("progress.csv", restore=args.restore)

    # prune by optuna when using distributed
    prune = torch.tensor(0, device=args.device)

    for epoch in range(args.start_epoch, args.epochs):
        timer = TimeMeter()

        # In distributed mode, calling the set_epoch method is needed to make shuffling
        # work; each process will use the same random seed otherwise.
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train
        loss, train_metrics = train(
            optimizer, model, train_loader, class_weight, epoch, args
        )
        if not args.distributed or args.rank == 0:
            timer.display(epoch, f"Epoch {epoch}, train")

        # bad, we get nan
        if np.isnan(loss):
            print("\n\nBad, we get nan for loss. Existing")
            sys.stdout.flush()
            sys.exit(1)

        # evaluate
        val_metrics = evaluate(model, val_loader, args)
        if not args.distributed or args.rank == 0:
            timer.display(epoch, f"Epoch {epoch}, evaluate")

        f1 = val_metrics.f1

        if stopper.step(-f1):
            break

        scheduler.step(-f1)

        is_best = f1 > best
        if is_best:
            best = f1

        # save checkpoint
        if not args.distributed or args.rank == 0:
            misc_objs = {"best": best, "epoch": epoch}
            save_checkpoints(
                state_dict_objs, misc_objs, is_best, msg=f"epoch: {epoch}, score {f1}"
            )

            _, epoch_time = timer.display(epoch, f"Epoch {epoch}, epoch time")

            stat = {"epoch": epoch, "loss": loss, "time": epoch_time}
            stat.update(train_metrics.as_dict("tr_metrics"))
            stat.update(val_metrics.as_dict("va_metrics"))

            progress.update(stat, save=True)
            progress.display()

        # Handle pruning based on the intermediate value.
        if not args.distributed:
            trial.report(best, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        else:
            if args.rank == 0:
                trial.report(best, epoch)

                if trial.should_prune():
                    # signal error by file
                    with open("score_for_optuna.txt", "w") as f:
                        f.write("prune")
                    prune = torch.tensor(1, device=args.device)

            # rank 0 signaled prune in the last epoch
            if args.distributed:
                dist.broadcast(prune, 0)
                if prune == 1:
                    return

    ################################################################################
    # test
    ################################################################################
    if not args.distributed or args.rank == 0:

        # load best to calculate test accuracy
        load_checkpoints(
            state_dict_objs, map_location=args.device, filename="best_checkpoint.pkl"
        )

        test_metrics = evaluate(model, test_loader, args)

        stat = test_metrics.as_dict("te_metrics")
        progress = ProgressMeter("test_result.csv")
        progress.update(stat, save=True)
        print("\nTest result:")
        progress.display()

        print(f"\nFinish training at: {datetime.now()}")

        # write best value for optuna
        with open("score_for_optuna.txt", "w") as f:
            f.write(str(best))

    return best


def create_optuna_study(filename="optuna.db"):
    """
    Create a optuna study using a sqlite database.

    Args:
        filename: path to the sqlite database.

    Returns:
        study
    """
    path = str(to_path(filename))

    study = optuna.create_study(
        storage=f"sqlite:///{path}",
        study_name="rxnrep_hyperparams",
        load_if_exists=True,
        direction="maximize",
    )

    return study


def copy_trial_value_to_args(trial, args):

    args.embedding_size = trial.suggest_categorical(
        "embedding_size", choices=[24, 36, 48]
    )

    num_molecule_conv_layers = trial.suggest_int("num_molecule_conv_layers", 2, 4)
    molecule_conv_layer_size = trial.suggest_categorical(
        "molecule_conv_layer_size", choices=[64, 128, 256]
    )
    args.molecule_conv_layer_sizes = [
        molecule_conv_layer_size
    ] * num_molecule_conv_layers

    return args


def objective(trial):
    args = parse_args()
    args = copy_trial_value_to_args(trial, args)

    if args.distributed:
        if args.world_size is None:
            raise RuntimeError("running distributed mode but `world_size` not set")
        mp.spawn(main, nprocs=args.world_size, args=(trial, args))

        # write best value for optuna
        with open("score_for_optuna.txt", "r") as f:
            line = f.readline().strip()
            if line == "prune":
                raise optuna.TrialPruned()
            else:
                score = float()

        # remove it so as not to use it the next time
        f = to_path("score_for_optuna.txt")
        f.unlink()

    else:
        score = main(None, trial, args)

    return score


if __name__ == "__main__":
    study = create_optuna_study(filename="optuna.db")
    study.optimize(objective, n_trials=2)

    # to run distributed CPU training:
    # python this_file_name.py --distributed 1 --world-size 2
