import logging
import warnings
from pathlib import Path

from torch.utils.data.dataloader import DataLoader

from rxnrep.data.electrolyte import ElectrolyteDataset
from rxnrep.data.featurizer import (
    AtomFeaturizer,
    AtomFeaturizerMinimum2,
    BondFeaturizer,
    BondFeaturizerMinimum,
    GlobalFeaturizer,
    MorganFeaturizer,
)
from rxnrep.data.green import (
    GreenClassicalFeaturesDataset,
    GreenClassificationDataset,
    GreenDataset,
)
from rxnrep.data.nrel import NRELDataset
from rxnrep.data.uspto import USPTOClassicalFeaturesDataset, USPTODataset

logger = logging.getLogger(__name__)


def load_dataset(args):
    if "schneider" in args.dataset or "tpl" in args.dataset:
        return load_uspto_dataset(args)
    elif "electrolyte" in args.dataset:
        return load_electrolyte_dataset(args)
    elif "green" in args.dataset:
        return load_green_dataset(args)

    else:
        raise ValueError(f"Not supported dataset {args.dataset}")


def load_green_dataset(args):
    classification = "classification" in args.dataset

    state_dict_filename = get_state_dict_filename(args)

    atom_featurizer_kwargs = {
        "atom_total_degree_one_hot": {"allowable_set": list(range(5))},
        "atom_total_valence_one_hot": {"allowable_set": list(range(5))},
        "atom_num_radical_electrons_one_hot": {"allowable_set": list(range(3))},
    }

    if args.reaction_conv_layer_sizes:
        build_reaction_graph = True
    else:
        build_reaction_graph = False

    if classification:
        DT = GreenClassificationDataset
        allow_label_scaler_none = None  # never used
    else:
        DT = GreenDataset
        allow_label_scaler_none = args.allow_label_scaler_none

    trainset = DT(
        filename=args.trainset_filename,
        atom_featurizer=AtomFeaturizer(featurizer_kwargs=atom_featurizer_kwargs),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        build_reaction_graph=build_reaction_graph,
        transform_features=True,
        init_state_dict=state_dict_filename,
        num_processes=args.nprocs,
        # label args
        allow_label_scaler_none=allow_label_scaler_none,
    )

    state_dict = trainset.state_dict()

    valset = DT(
        filename=args.valset_filename,
        atom_featurizer=AtomFeaturizer(featurizer_kwargs=atom_featurizer_kwargs),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        build_reaction_graph=build_reaction_graph,
        transform_features=True,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        # label args
        allow_label_scaler_none=allow_label_scaler_none,
    )

    testset = DT(
        filename=args.testset_filename,
        atom_featurizer=AtomFeaturizer(featurizer_kwargs=atom_featurizer_kwargs),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        build_reaction_graph=build_reaction_graph,
        transform_features=True,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        # label args
        allow_label_scaler_none=allow_label_scaler_none,
    )

    train_loader, val_loader, test_loader = _get_loaders(
        trainset, valset, testset, args.batch_size, args.num_workers
    )

    # TODO move this out the function? It's hard to know what's going on if we set args
    #  all the time
    # Add info that will be used in the model to args for easy access
    args.feature_size = trainset.feature_size

    if classification:
        class_weight = trainset.get_class_weight(
            num_reaction_classes=args.num_reaction_classes, class_weight_as_1=True
        )
        args.reaction_class_weight = class_weight["reaction_type"]
    else:
        args.label_scaler = trainset.get_label_scaler()

    # save dataset state dict for retraining or prediction
    trainset.save_state_dict_file(args.dataset_state_dict_filename)
    logger.info(
        f"Trainset size: {len(trainset)}, valset size: {len(valset)}: "
        f"testset size: {len(testset)}."
    )

    return train_loader, val_loader, test_loader


def load_uspto_dataset(args):

    state_dict_filename = get_state_dict_filename(args)

    has_class_label = args.has_class_label if "has_class_label" in args else False

    if args.reaction_conv_layer_sizes:
        build_reaction_graph = True
    else:
        build_reaction_graph = False

    trainset = USPTODataset(
        filename=args.trainset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        build_reaction_graph=build_reaction_graph,
        init_state_dict=state_dict_filename,
        num_processes=args.nprocs,
        transform_features=True,
        # label args
        has_class_label=has_class_label,
    )

    state_dict = trainset.state_dict()

    valset = USPTODataset(
        filename=args.valset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        build_reaction_graph=build_reaction_graph,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        transform_features=True,
        # label args
        has_class_label=has_class_label,
    )

    testset = USPTODataset(
        filename=args.testset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        build_reaction_graph=build_reaction_graph,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        transform_features=True,
        # label args
        has_class_label=has_class_label,
    )

    train_loader, val_loader, test_loader = _get_loaders(
        trainset, valset, testset, args.batch_size, args.num_workers
    )

    # Add info that will be used in the model to args for easy access
    args.feature_size = trainset.feature_size

    if has_class_label:
        class_weight = trainset.get_class_weight(
            num_reaction_classes=args.num_reaction_classes, class_weight_as_1=True
        )
        args.reaction_class_weight = class_weight["reaction_type"]

    # save dataset state dict for retraining or prediction
    trainset.save_state_dict_file(args.dataset_state_dict_filename)
    logger.info(
        f"Trainset size: {len(trainset)}, valset size: {len(valset)}: "
        f"testset size: {len(testset)}."
    )

    return train_loader, val_loader, test_loader


def load_electrolyte_dataset(args):
    state_dict_filename = get_state_dict_filename(args)

    if args.reaction_conv_layer_sizes:
        build_reaction_graph = True
    else:
        build_reaction_graph = False

    trainset = ElectrolyteDataset(
        filename=args.trainset_filename,
        atom_featurizer=AtomFeaturizerMinimum2(),
        bond_featurizer=BondFeaturizerMinimum(),
        global_featurizer=GlobalFeaturizer(allowable_charge=[-1, 0, 1]),
        build_reaction_graph=build_reaction_graph,
        init_state_dict=state_dict_filename,
        num_processes=args.nprocs,
        transform_features=True,
        # label args
        allow_label_scaler_none=args.allow_label_scaler_none,
    )

    state_dict = trainset.state_dict()

    valset = ElectrolyteDataset(
        filename=args.valset_filename,
        atom_featurizer=AtomFeaturizerMinimum2(),
        bond_featurizer=BondFeaturizerMinimum(),
        global_featurizer=GlobalFeaturizer(allowable_charge=[-1, 0, 1]),
        build_reaction_graph=build_reaction_graph,
        transform_features=True,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        # label args
        allow_label_scaler_none=args.allow_label_scaler_none,
    )

    testset = ElectrolyteDataset(
        filename=args.testset_filename,
        atom_featurizer=AtomFeaturizerMinimum2(),
        bond_featurizer=BondFeaturizerMinimum(),
        global_featurizer=GlobalFeaturizer(allowable_charge=[-1, 0, 1]),
        build_reaction_graph=build_reaction_graph,
        transform_features=True,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        # label args
        allow_label_scaler_none=args.allow_label_scaler_none,
    )

    train_loader, val_loader, test_loader = _get_loaders(
        trainset, valset, testset, args.batch_size, args.num_workers
    )

    # Add info that will be used in the model to args for easy access
    args.feature_size = trainset.feature_size
    args.label_scaler = trainset.get_label_scaler()

    # save dataset state dict for retraining or prediction
    trainset.save_state_dict_file(args.dataset_state_dict_filename)
    logger.info(
        f"Trainset size: {len(trainset)}, valset size: {len(valset)}: "
        f"testset size: {len(testset)}."
    )

    return train_loader, val_loader, test_loader


def load_nrel_dataset(args):

    state_dict_filename = get_state_dict_filename(args)

    if args.reaction_conv_layer_sizes:
        build_reaction_graph = True
    else:
        build_reaction_graph = False

    trainset = NRELDataset(
        filename=args.trainset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        build_reaction_graph=build_reaction_graph,
        transform_features=True,
        init_state_dict=state_dict_filename,
        num_processes=args.nprocs,
        allow_label_scaler_none=args.allow_label_scaler_none,
    )

    state_dict = trainset.state_dict()

    valset = NRELDataset(
        filename=args.valset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        build_reaction_graph=build_reaction_graph,
        transform_features=True,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        allow_label_scaler_none=args.allow_label_scaler_none,
    )

    testset = NRELDataset(
        filename=args.testset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        build_reaction_graph=build_reaction_graph,
        transform_features=True,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        allow_label_scaler_none=args.allow_label_scaler_none,
    )

    train_loader, val_loader, test_loader = _get_loaders(
        trainset, valset, testset, args.batch_size, args.num_workers
    )

    # Add info that will be used in the model to args for easy access
    args.feature_size = trainset.feature_size

    args.label_scaler = trainset.get_label_scaler()

    # save dataset state dict for retraining or prediction
    trainset.save_state_dict_file(args.dataset_state_dict_filename)
    logger.info(
        f"Trainset size: {len(trainset)}, valset size: {len(valset)}: "
        f"testset size: {len(testset)}."
    )

    return train_loader, val_loader, test_loader


def load_morgan_feature_dataset(args):

    featurizer = MorganFeaturizer(
        radius=args.morgan_radius,
        size=args.morgan_size,
    )

    if "green" in args.dataset:
        DT = GreenClassicalFeaturesDataset
    elif "schneider" in args.dataset or "tpl" in args.dataset:
        DT = USPTOClassicalFeaturesDataset
    else:
        raise ValueError(f"Not supported dataset {args.dataset}")

    trainset = DT(
        filename=args.trainset_filename,
        featurizer=featurizer,
        feature_type=args.feature_pool_type,
        num_processes=args.nprocs,
    )
    valset = DT(
        filename=args.valset_filename,
        featurizer=featurizer,
        feature_type=args.feature_pool_type,
        num_processes=args.nprocs,
    )
    testset = DT(
        filename=args.testset_filename,
        featurizer=featurizer,
        feature_type=args.feature_pool_type,
        num_processes=args.nprocs,
    )

    logger.info(
        f"Trainset size: {len(trainset)}, valset size: {len(valset)}: "
        f"testset size: {len(testset)}."
    )

    return _get_loaders(trainset, valset, testset, args.batch_size, args.num_workers)


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


def _get_loaders(trainset, valset, testset, batch_size, num_workers):

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
