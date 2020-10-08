import sys
import time
import warnings
import torch
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from rxnrep.data.uspto import USPTODataset, collate_fn
from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.data.splitter import train_validation_test_split
from rxnrep.model.model import ReactionRepresentation
from rxnrep.model.metric import MultiClassificationMetrics, BinaryClassificationMetrics
from rxnrep.model.clustering import ReactionCluster, DistributedReactionCluster
from rxnrep.scripts.utils import (
    EarlyStopping,
    seed_all,
    load_checkpoints,
    save_checkpoints,
)
from rxnrep.utils import yaml_dump
from rxnrep.scripts.utils import init_distributed_mode

best = -np.finfo(np.float32).max


def parse_args():
    parser = argparse.ArgumentParser(description="Reaction Representation")

    # TODO, for temporary test only
    # ========== input files ==========
    fname = "/Users/mjwen/Documents/Dataset/uspto/raw/2001_Sep2016_USPTOapplications_smiles_n200_processed.tsv"
    parser.add_argument("--dataset-filename", type=str, default=fname)

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

    # ========== decoder ==========
    parser.add_argument(
        "--decoder-hidden-layer-sizes", type=int, nargs="+", default=[64, 64]
    )
    parser.add_argument("--decoder-activation", type=str, default="ReLU")

    # ========== training ==========
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--restore", type=int, default=0, help="restore training")
    parser.add_argument(
        "--dataset-state-dict-filename", type=str, default="dataset_state_dict.yaml"
    )

    # ========== distributed ==========
    parser.add_argument("--gpu", type=int, default=0, help="Whether to use GPU.")
    parser.add_argument(
        "--distributed", type=int, default=0, help="Whether distributed DDP training.",
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://localhost:13456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--local_rank", type=int, help="Local rank of process.")

    args = parser.parse_args()

    return args


def train(optimizer, model, data_loader, reaction_cluster, epoch, device):

    model.train()

    nodes = ["atom", "bond", "global"]

    metrics = {
        "bond_type": MultiClassificationMetrics(num_classes=3),
        "atom_in_reaction_center": BinaryClassificationMetrics(),
    }

    # TODO temporary (should set to the ratio of atoms in center)
    pos_weight = torch.tensor(4.0).to(device)

    cluster_labels = reaction_cluster.get_cluster_assignments()
    # TODO temporary, use the first head for now
    cluster_labels = cluster_labels[0]

    epoch_loss = 0.0
    for it, (indices, mol_graphs, rxn_graphs, labels, metadata) in enumerate(data_loader):
        mol_graphs = mol_graphs.to(device)
        rxn_graphs = rxn_graphs.to(device)
        feats = {nt: mol_graphs.nodes[nt].data.pop("feat").to(device) for nt in nodes}
        labels = {k: v.to(device) for k, v in labels.items()}

        preds, rxn_embeddings = model(mol_graphs, rxn_graphs, feats, metadata)
        preds["atom_in_reaction_center"] = torch.flatten(preds["atom_in_reaction_center"])

        # TODO may be assign different weights for atoms and bonds, giving each
        #  reaction have the same weight?
        loss_bond_type = F.cross_entropy(
            preds["bond_type"], labels["bond_type"], reduction="mean"
        )
        loss_atom_in_reaction_center = F.binary_cross_entropy_with_logits(
            preds["atom_in_reaction_center"],
            labels["atom_in_reaction_center"],
            reduction="mean",
            pos_weight=pos_weight,
        )

        # @@@
        # TODO temporaty, random label
        x = len(preds["reaction_cluster"])
        cluster_labels = cluster_labels[:x]
        # @@@
        loss_reaction_cluster = F.cross_entropy(preds["reaction_cluster"], cluster_labels)
        loss = loss_bond_type + loss_atom_in_reaction_center + loss_reaction_cluster

        # update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

        # metrics
        # bond type
        p = torch.argmax(preds["bond_type"], dim=1)
        metrics["bond_type"].step(p, labels["bond_type"])
        # atom in reaction center
        p = torch.sigmoid(preds["atom_in_reaction_center"]) > 0.5
        metrics["atom_in_reaction_center"].step(p, labels["atom_in_reaction_center"])

    epoch_loss /= it + 1

    # compute metric values
    metrics["bond_type"].compute_metric_values(class_reduction="weighted")
    metrics["atom_in_reaction_center"].compute_metric_values()

    return epoch_loss, metrics


def evaluate(model, data_loader, device):
    model.eval()

    nodes = ["atom", "bond", "global"]

    metrics = {
        "bond_type": MultiClassificationMetrics(num_classes=3),
        "atom_in_reaction_center": BinaryClassificationMetrics(),
    }

    with torch.no_grad():

        for it, (indices, mol_graphs, rxn_graphs, labels, metadata) in enumerate(
            data_loader
        ):
            mol_graphs = mol_graphs.to(device)
            rxn_graphs = rxn_graphs.to(device)
            feats = {nt: mol_graphs.nodes[nt].data.pop("feat").to(device) for nt in nodes}
            labels = {k: v.to(device) for k, v in labels.items()}

            preds, rxn_embeddings = model(mol_graphs, rxn_graphs, feats, metadata)
            preds["atom_in_reaction_center"] = torch.flatten(
                preds["atom_in_reaction_center"]
            )

            # metrics
            # bond type
            p = torch.argmax(preds["bond_type"], dim=1)
            metrics["bond_type"].step(p, labels["bond_type"])
            # atom in reaction center
            p = torch.sigmoid(preds["atom_in_reaction_center"]) > 0.5
            metrics["atom_in_reaction_center"].step(p, labels["atom_in_reaction_center"])

    # compute metric values
    metrics["bond_type"].compute_metric_values(class_reduction="weighted")
    metrics["atom_in_reaction_center"].compute_metric_values()

    return metrics


def load_dataset(args, validation_ratio=0.1, test_ratio=0.1):

    # check dataset state dict if restore model
    if args.restore:
        if args.dataset_state_dict_filename is None:
            warnings.warn("Restore with `args.dataset_state_dict_filename` set to None.")
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

    dataset = USPTODataset(
        filename=args.dataset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        state_dict_filename=state_dict_filename,
    )

    # trainset, valset, testset = train_validation_test_split(
    #     dataset, validation=validation_ratio, test=test_ratio
    # )

    # TODO, train set, val set, and test set should be separate dataset, since we will
    #  directly use index in clustering
    trainset = dataset
    valset = dataset
    testset = dataset

    # save dataset state dict for retraining or prediction
    if not args.distributed or args.rank == 0:
        dataset.save_state_dict(args.dataset_state_dict_filename)
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
        collate_fn=collate_fn,
    )
    # larger val and test set batch_size is faster but needs more memory
    # adjust the batch size of to fit memory
    bs = max(len(valset) // 10, 1)
    val_loader = DataLoader(valset, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    bs = max(len(testset) // 10, 1)
    test_loader = DataLoader(testset, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    # atom, bond, and global feature size
    feature_size = dataset.feature_size

    return train_loader, val_loader, test_loader, feature_size, train_sampler, trainset


def main(args):
    # TODO no need to make best global, since now each process will have all the val data
    global best

    if args.distributed:
        init_distributed_mode(args)
    else:
        if args.gpu:
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cpu")

    if not args.distributed or args.rank == 0:
        print(args)
        yaml_dump(args, "train_args.yaml")
        print("\n\nStart training at:", datetime.now())

    # Explicitly setting seed to ensure the same dataset split and models created in
    # two processes (when distributed) starting from the same random weights and biases
    seed_all()

    ################################################################################
    #  set up dataset, model, optimizer ...
    ################################################################################

    ### dataset
    (
        train_loader,
        val_loader,
        test_loader,
        feats_size,
        train_sampler,
        trainset,
    ) = load_dataset(args)

    ### model
    model = ReactionRepresentation(
        in_feats=feats_size,
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
        # bond type decoder
        bond_type_decoder_hidden_layer_sizes=args.decoder_hidden_layer_sizes,
        bond_type_decoder_activation=args.decoder_activation,
        # atom in reaction center decoder
        atom_in_reaction_center_decoder_hidden_layer_sizes=args.decoder_hidden_layer_sizes,
        atom_in_reaction_center_decoder_activation=args.decoder_activation,
        # clustering decoder
        reaction_cluster_decoder_hidden_layer_sizes=args.decoder_hidden_layer_sizes,
        reaction_cluster_decoder_activation=args.decoder_activation,
        num_prototypes=10,
    )

    if not args.distributed or args.rank == 0:
        print(model)

    if args.distributed:
        if args.gpu:
            model.to(args.device)
            ddp_model = DDP(model, device_ids=[args.device])
        else:
            ddp_model = DDP(model)
        model = ddp_model

    ### optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    ### learning rate scheduler and stopper
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.4, patience=50, verbose=True
    )
    stopper = EarlyStopping(patience=150)

    ### cluster
    if args.distributed:
        reaction_cluster = DistributedReactionCluster(
            model,
            train_loader,
            args.batch_size,
            len(train_loader.dataset),
            num_prototypes=[10],
            device=args.device,
        )
    else:
        reaction_cluster = ReactionCluster(
            model, trainset, num_clusters=10, device=args.device
        )

    # load checkpoint
    state_dict_objs = {"model": model, "optimizer": optimizer, "scheduler": scheduler}
    if args.restore:
        try:
            checkpoint = load_checkpoints(
                state_dict_objs, map_location=args.device, filename="checkpoint.pkl"
            )

            args.start_epoch = checkpoint["epoch"]
            best = checkpoint["best"]
            print(f"Successfully load checkpoints, best {best}, epoch {args.start_epoch}")

        except FileNotFoundError as e:
            warnings.warn(str(e) + " Continue without loading checkpoints.")
            pass

    ################################################################################
    # training loop
    ################################################################################

    if not args.distributed or args.rank == 0:
        print(
            "\n\n# Epoch     Loss      Train[acc|prec|rec|f1]    "
            "Val[acc|prec|rec|f1]   Time"
        )
        sys.stdout.flush()

    for epoch in range(args.start_epoch, args.epochs):
        ti = time.time()

        # In distributed mode, calling the set_epoch method is needed to make shuffling
        # work; each process will use the same random seed otherwise.
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train
        loss, train_metrics = train(
            optimizer, model, train_loader, reaction_cluster, epoch, args.device
        )

        # bad, we get nan
        if np.isnan(loss):
            print("\n\nBad, we get nan for loss. Existing")
            sys.stdout.flush()
            sys.exit(1)

        # evaluate
        val_metrics = evaluate(model, val_loader, args.device)

        f1_sum = val_metrics["bond_type"].f1 + val_metrics["atom_in_reaction_center"].f1

        if stopper.step(-f1_sum):
            break

        scheduler.step(f1_sum)

        is_best = f1_sum > best
        if is_best:
            best = f1_sum

        # save checkpoint
        if not args.distributed or args.rank == 0:
            misc_objs = {"best": best, "epoch": epoch}
            save_checkpoints(
                state_dict_objs,
                misc_objs,
                is_best,
                msg=f"epoch: {epoch}, score {f1_sum}",
            )

            tt = time.time() - ti

            print(
                "{:5d}   {:12.6e}   {}   {}   {}   {}   {:.2f}".format(
                    epoch,
                    loss,
                    str(train_metrics["bond_type"]),
                    str(train_metrics["atom_in_reaction_center"]),
                    str(val_metrics["bond_type"]),
                    str(val_metrics["atom_in_reaction_center"]),
                    tt,
                )
            )
            if epoch % 10 == 0:
                sys.stdout.flush()

    ################################################################################
    # test
    ################################################################################

    # load best to calculate test accuracy
    load_checkpoints(
        state_dict_objs, map_location=args.device, filename="best_checkpoint.pkl"
    )

    if not args.distributed or args.rank == 0:
        # test_metrics = evaluate(model, test_loader, args.device)
        test_metrics = evaluate(model, val_loader, args.device)

        print(f"\n#Test Metric (bond_type): {str(test_metrics['bond_type'])}")
        print(
            "\n#Test Metric (atom_in_reaction_center): "
            f"{str(test_metrics['atom_in_reaction_center'])}"
        )
        print(f"\nFinish training at: {datetime.now()}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

    # to run distributed CPU training, do
    # python -m torch.distributed.launch --nproc_per_node=2 train_distributed.py  --distributed 1
