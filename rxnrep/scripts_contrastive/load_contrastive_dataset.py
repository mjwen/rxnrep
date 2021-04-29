import logging
import warnings
from pathlib import Path

from torch.utils.data.dataloader import DataLoader

from rxnrep.data import transforms
from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.data.green import GreenContrastiveDataset
from rxnrep.data.uspto import USPTOContrastiveDataset

logger = logging.getLogger(__name__)


def load_dataset(args):
    if "schneider" in args.dataset or "tpl" in args.dataset:
        return load_uspto_dataset(args)
    elif args.dataset == "green":
        return load_green_dataset(args)
    else:
        raise ValueError(f"Not supported dataset {args.dataset}")


def load_uspto_dataset(args):

    state_dict_filename = get_state_dict_filename(args)

    atom_featurizer = AtomFeaturizer()
    bond_featurizer = BondFeaturizer()
    global_featurizer = GlobalFeaturizer()

    if args.reaction_conv_layer_sizes:
        build_reaction_graph = True
    else:
        build_reaction_graph = False

    t1, t2 = init_augmentations(args)

    trainset = USPTOContrastiveDataset(
        filename=args.trainset_filename,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        build_reaction_graph=build_reaction_graph,
        init_state_dict=state_dict_filename,
        num_processes=args.nprocs,
        transform_features=True,
        transform1=t1,
        transform2=t2,
    )
    state_dict = trainset.state_dict()

    valset = USPTOContrastiveDataset(
        filename=args.valset_filename,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        build_reaction_graph=build_reaction_graph,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        transform_features=True,
        transform1=t1,
        transform2=t2,
    )

    testset = USPTOContrastiveDataset(
        filename=args.testset_filename,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        build_reaction_graph=build_reaction_graph,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        transform_features=True,
        transform1=t1,
        transform2=t2,
    )

    # save dataset state dict for retraining or prediction
    trainset.save_state_dict_file(args.dataset_state_dict_filename)
    logger.info(
        f"Trainset size: {len(trainset)}, valset size: {len(valset)}: "
        f"testset size: {len(testset)}."
    )

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trainset.collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valset.collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=testset.collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Add info that will be used in the model to args for easy access
    args.feature_size = trainset.feature_size

    return train_loader, val_loader, test_loader


def load_green_dataset(args):

    state_dict_filename = get_state_dict_filename(args)

    atom_featurizer_kwargs = {
        "atom_total_degree_one_hot": {"allowable_set": list(range(5))},
        "atom_total_valence_one_hot": {"allowable_set": list(range(5))},
        "atom_num_radical_electrons_one_hot": {"allowable_set": list(range(3))},
    }
    atom_featurizer = AtomFeaturizer(featurizer_kwargs=atom_featurizer_kwargs)
    bond_featurizer = BondFeaturizer()
    global_featurizer = GlobalFeaturizer()

    if args.reaction_conv_layer_sizes:
        build_reaction_graph = True
    else:
        build_reaction_graph = False

    t1, t2 = init_augmentations(args)

    trainset = GreenContrastiveDataset(
        filename=args.trainset_filename,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        build_reaction_graph=build_reaction_graph,
        init_state_dict=state_dict_filename,
        num_processes=args.nprocs,
        transform_features=True,
        transform1=t1,
        transform2=t2,
    )

    state_dict = trainset.state_dict()

    valset = GreenContrastiveDataset(
        filename=args.valset_filename,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        build_reaction_graph=build_reaction_graph,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        transform_features=True,
        transform1=t1,
        transform2=t2,
    )

    testset = GreenContrastiveDataset(
        filename=args.testset_filename,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        build_reaction_graph=build_reaction_graph,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        transform_features=True,
        transform1=t1,
        transform2=t2,
    )

    # save dataset state dict for retraining or prediction
    trainset.save_state_dict_file(args.dataset_state_dict_filename)
    logger.info(
        f"Trainset size: {len(trainset)}, valset size: {len(valset)}: "
        f"testset size: {len(testset)}."
    )

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trainset.collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valset.collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=testset.collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Add info that will be used in the model to args for easy access
    args.feature_size = trainset.feature_size

    return train_loader, val_loader, test_loader


def get_state_dict_filename(args):
    """
    Check dataset state dict if in restore mode
    """

    # finetune mode
    if "pretrained_dataset_state_dict_filename" in args:
        if not Path(args.pretrained_dataset_state_dict_filename).exists():
            raise ValueError(
                f"args.pretrained_dataset_state_dict_filename: "
                f"`{args.pretrained_dataset_state_dict_filename}` not found."
            )
        else:
            state_dict_filename = args.pretrained_dataset_state_dict_filename

    else:
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

    return state_dict_filename


def init_augmentations(args):
    def select_transform(
        name,
        ratio,
        select_mode,
        center_mode,
        ratio_multiplier,
        mask_value_atom,
        mask_value_bond,
        functional_group_smarts_files,
    ):

        if name == "drop_atom":
            t = transforms.DropAtom(ratio, select_mode, ratio_multiplier)

        elif name == "drop_bond":
            t = transforms.DropBond(ratio, select_mode, ratio_multiplier)

        elif name == "mask_atom":
            t = transforms.MaskAtomAttribute(
                ratio, select_mode, ratio_multiplier, mask_value=mask_value_atom
            )

        elif name == "mask_bond":
            t = transforms.MaskBondAttribute(
                ratio, select_mode, ratio_multiplier, mask_value=mask_value_bond
            )

        elif name == "subgraph":

            t = transforms.Subgraph(
                ratio,
                select_mode,
                ratio_multiplier,
                center_mode,
                functional_group_smarts_files,
            )

        elif name == "subgraph_bfs":
            t = transforms.SubgraphBFS(ratio, select_mode, ratio_multiplier)

        elif name == "identity":
            t = transforms.IdentityTransform(ratio, select_mode, ratio_multiplier)

        elif name == "subgraph_or_identity":
            t1 = transforms.Subgraph(
                ratio,
                select_mode,
                ratio_multiplier,
                center_mode,
                functional_group_smarts_files,
            )
            t2 = transforms.IdentityTransform(ratio, select_mode, ratio_multiplier)
            t = transforms.OneOrTheOtherTransform(t1, t2, first_probability=0.5)

        elif name == "subgraph_bfs_or_identity":
            t1 = transforms.SubgraphBFS(ratio, select_mode, ratio_multiplier)
            t2 = transforms.IdentityTransform(ratio, select_mode, ratio_multiplier)
            t = transforms.OneOrTheOtherTransform(t1, t2, first_probability=0.5)

        else:
            raise ValueError(f"Unsupported augmentation type {name}")

        return t

    t1 = select_transform(
        args.augment_1,
        args.augment_1_ratio,
        args.augment_1_select_mode,
        args.augment_1_center_mode,
        args.augment_1_ratio_multiplier,
        args.augment_mask_value_atom,
        args.augment_mask_value_bond,
        args.augment_functional_group_smarts_files,
    )
    t2 = select_transform(
        args.augment_2,
        args.augment_2_ratio,
        args.augment_2_select_mode,
        args.augment_2_center_mode,
        args.augment_2_ratio_multiplier,
        args.augment_mask_value_atom,
        args.augment_mask_value_bond,
        args.augment_functional_group_smarts_files,
    )

    return t1, t2
