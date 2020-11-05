import sys
import warnings
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from rxnrep.data.electrolyte import ElectrolyteDataset
from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
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
from rxnrep.scripts.utils import init_distributed_mode, ProgressMeter, TimeMeter

best = -np.finfo(np.float32).max


def parse_args():
    parser = argparse.ArgumentParser(description="Reaction Representation")

    # ========== input files ==========
    prefix = "/Users/mjwen/Documents/Dataset/rxnrep/"

    fname_tr = prefix + "reactions_n2000.json"
    fname_val = prefix + "reactions_n2000.json"
    fname_test = prefix + "reactions_n2000.json"

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

    # ========== decoder ==========
    parser.add_argument(
        "--decoder-hidden-layer-sizes", type=int, nargs="+", default=[64, 64]
    )
    parser.add_argument("--decoder-activation", type=str, default="ReLU")
    # clustering decoder
    parser.add_argument(
        "--cluster-decoder-projection-head-size",
        type=int,
        default=33,
        help="projection head size for the clustering decoder",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature in the loss for cluster decoder",
    )

    # ========== training ==========
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--restore", type=int, default=0, help="restore training")
    parser.add_argument(
        "--dataset-state-dict-filename", type=str, default="dataset_state_dict.yaml"
    )

    # ========== distributed ==========
    parser.add_argument(
        "--launch-mode",
        type=str,
        default="torch_launch",
        help="How to launch distributed training: [`torch_launch`| `srun`]",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Whether to use GPU.")
    parser.add_argument(
        "--distributed", type=int, default=0, help="Whether distributed DDP training.",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        help="tcp port for distributed communication e.g. `tcp://localhost:15678`.",
    )
    parser.add_argument("--local_rank", type=int, help="Local rank of process.")

    args = parser.parse_args()

    return args


def train(optimizer, model, data_loader, reaction_cluster, class_weights, epoch, args):
    timer = TimeMeter(frequency=5)

    model.train()

    nodes = ["atom", "bond", "global"]

    # class weights
    atom_in_center_weight = class_weights["atom_in_reaction_center"].to(args.device)
    bond_type_weight = class_weights["bond_type"].to(args.device)

    if not args.distributed or args.rank == 0:
        timer.display(epoch, f"In epoch; class weight")

    # evaluation metrics
    metrics = {
        "bond_type": MultiClassificationMetrics(num_classes=3),
        "atom_in_reaction_center": BinaryClassificationMetrics(),
    }

    # cluster to get assignments and centroids
    assignments, centroids = reaction_cluster.get_cluster_assignments()

    if not args.distributed or args.rank == 0:
        timer.display(epoch, f"In epoch; clustering")

    # keep track of the data and index to be used in the next clustering run (to same
    # time)
    data_for_cluster = []
    index_for_cluster = []

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
        labels = {k: v.to(args.device) for k, v in labels.items()}

        if not args.distributed or args.rank == 0:
            timer.display(it, f"Batch {it}; to cuda")

        preds, rxn_embeddings = model(mol_graphs, rxn_graphs, feats, metadata)

        if not args.distributed or args.rank == 0:
            timer.display(it, f"Batch {it}; model predict")

        # ========== loss for bond type prediction ==========
        loss_bond_type = F.cross_entropy(
            preds["bond_type"],
            labels["bond_type"],
            reduction="mean",
            weight=bond_type_weight,
        )

        if not args.distributed or args.rank == 0:
            timer.display(it, f"Batch {it}; bond type prediction loss")

        # ========== loss for atom in reaction center prediction ==========
        preds["atom_in_reaction_center"] = preds["atom_in_reaction_center"].flatten()
        loss_atom_in_reaction_center = F.binary_cross_entropy_with_logits(
            preds["atom_in_reaction_center"],
            labels["atom_in_reaction_center"],
            reduction="mean",
            pos_weight=atom_in_center_weight,
        )

        if not args.distributed or args.rank == 0:
            timer.display(it, f"Batch {it}; reaction center loss")

        # ========== loss for clustering prediction ==========
        loss_reaction_cluster = []
        for a, c in zip(assignments, centroids):
            a = a[indices].to(args.device)  # select for current batch from all
            c = c.to(args.device)
            p = torch.mm(preds["reaction_cluster"], c.t()) / args.temperature
            e = F.cross_entropy(p, a)
            loss_reaction_cluster.append(e)
        loss_reaction_cluster = sum(loss_reaction_cluster) / len(loss_reaction_cluster)

        if not args.distributed or args.rank == 0:
            timer.display(it, f"Batch {it}; clustering loss")

        # total loss
        # TODO may be assign different weights for atoms and bonds, giving each
        #  reaction have the same weight?
        loss = loss_bond_type + loss_atom_in_reaction_center + loss_reaction_cluster

        # ========== update model parameters ==========
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

        if not args.distributed or args.rank == 0:
            timer.display(it, f"Batch {it}; back propagation")

        # ========== metrics ==========
        # bond type
        p = torch.argmax(preds["bond_type"], dim=1)
        metrics["bond_type"].step(p, labels["bond_type"])
        # atom in reaction center
        p = torch.sigmoid(preds["atom_in_reaction_center"]) > 0.5
        p = p.to(torch.int32)
        metrics["atom_in_reaction_center"].step(p, labels["atom_in_reaction_center"])

        # ========== keep track of data ==========
        data_for_cluster.append(preds["reaction_cluster"].detach())
        index_for_cluster.append(indices.to(args.device))

        if not args.distributed or args.rank == 0:
            timer.display(it, f"Batch {it}; keep data for metric and clustering")

    if not args.distributed or args.rank == 0:
        timer.display(epoch, f"In epoch; Finish looping batch")

    # compute metric values
    epoch_loss /= it + 1
    metrics["bond_type"].compute_metric_values(class_reduction="weighted")
    metrics["atom_in_reaction_center"].compute_metric_values()

    if not args.distributed or args.rank == 0:
        timer.display(epoch, f"In epoch; Compute metric")

    # keep track of clustering data to be used in the next iteration
    reaction_cluster.set_local_data_and_index(
        torch.cat(data_for_cluster), torch.cat(index_for_cluster)
    )

    return epoch_loss, metrics


def evaluate(model, data_loader, args):
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
            mol_graphs = mol_graphs.to(args.device)
            rxn_graphs = rxn_graphs.to(args.device)
            feats = {
                nt: mol_graphs.nodes[nt].data.pop("feat").to(args.device)
                for nt in nodes
            }
            labels = {k: v.to(args.device) for k, v in labels.items()}

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
            metrics["atom_in_reaction_center"].step(
                p, labels["atom_in_reaction_center"]
            )

    # compute metric values
    metrics["bond_type"].compute_metric_values(class_reduction="weighted")
    metrics["atom_in_reaction_center"].compute_metric_values()

    return metrics


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

    trainset = ElectrolyteDataset(
        filename=args.trainset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=state_dict_filename,
    )
    valset = ElectrolyteDataset(
        filename=args.valset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=trainset.state_dict(),
    )
    testset = ElectrolyteDataset(
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
    )

    # TODO, for val set, we can also make it distributed and report the error on rank
    #  0. If the vl set size is large enough, the statistics of error should be the same
    #  in different ranks. If this is not good, we can gather and reduce the validation
    #  metric.

    # larger val and test set batch_size is faster but needs more memory
    # adjust the batch size of to fit memory
    bs = max(len(valset) // 10, 1)
    val_loader = DataLoader(
        valset, batch_size=bs, shuffle=False, collate_fn=valset.collate_fn
    )
    bs = max(len(testset) // 10, 1)
    test_loader = DataLoader(
        testset, batch_size=bs, shuffle=False, collate_fn=testset.collate_fn
    )

    return train_loader, val_loader, test_loader, train_sampler


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
    train_loader, val_loader, test_loader, train_sampler = load_dataset(args)

    ### model
    model = ReactionRepresentation(
        in_feats=train_loader.dataset.feature_size,
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
        reaction_cluster_decoder_output_size=args.cluster_decoder_projection_head_size,
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
    else:
        if args.gpu:
            model.to(args.device)

    ### optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    ### learning rate scheduler and stopper
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.4, patience=50, verbose=True
    )
    stopper = EarlyStopping(patience=150)

    ### prepare class weight for classification tasks
    (
        atom_in_reaction_center_weight,
        bond_type_weight,
    ) = train_loader.dataset.get_atom_in_reaction_center_and_bond_type_class_weight()
    class_weights = {
        "atom_in_reaction_center": atom_in_reaction_center_weight,
        "bond_type": bond_type_weight,
    }

    ### cluster
    if args.distributed:
        reaction_cluster = DistributedReactionCluster(
            model,
            train_loader,
            args.batch_size,
            num_centroids=[22, 22],
            device=args.device,
        )
    else:
        trainset = train_loader.dataset
        reaction_cluster = ReactionCluster(
            model, trainset, args.batch_size, num_centroids=[22, 22], device=args.device
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

    for epoch in range(args.start_epoch, args.epochs):
        timer = TimeMeter()

        # In distributed mode, calling the set_epoch method is needed to make shuffling
        # work; each process will use the same random seed otherwise.
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train
        loss, train_metrics = train(
            optimizer, model, train_loader, reaction_cluster, class_weights, epoch, args
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

            _, epoch_time = timer.display(epoch, f"Epoch {epoch}, epoch time")
            stat = {"epoch": epoch, "loss": loss, "time": epoch_time}
            stat.update(train_metrics["bond_type"].as_dict("tr_bt"))
            stat.update(train_metrics["atom_in_reaction_center"].as_dict("tr_airc"))
            progress.update(stat, save=True)
            progress.display()

    ################################################################################
    # test
    ################################################################################
    if not args.distributed or args.rank == 0:

        # load best to calculate test accuracy
        load_checkpoints(
            state_dict_objs, map_location=args.device, filename="best_checkpoint.pkl"
        )

        test_metrics = evaluate(model, test_loader, args)

        stat = test_metrics["bond_type"].as_dict("tr_bt")
        stat.update(test_metrics["atom_in_reaction_center"].as_dict("tr_airc"))

        progress = ProgressMeter("test_result.csv")
        progress.update(stat, save=True)
        print("\nTest result:")
        progress.display()

        print(f"\nFinish training at: {datetime.now()}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

    # to run distributed CPU training, do
    # python -m torch.distributed.launch --nproc_per_node=2 train_distributed.py  --distributed 1
