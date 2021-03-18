import logging
import warnings
from pathlib import Path

from torch.utils.data.dataloader import DataLoader

from rxnrep.data import transforms
from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.data.uspto import USPTOConstrativeDataset

logger = logging.getLogger(__name__)


def load_dataset(args):
    if "schneider" in args.dataset:
        return load_uspto_dataset(args)
    else:
        raise ValueError(f"Not supported dataset {args.dataset}")


def load_uspto_dataset(args):

    state_dict_filename = get_state_dict_filename(args)

    atom_featurizer = AtomFeaturizer()
    bond_featurizer = BondFeaturizer()
    global_featurizer = GlobalFeaturizer()

    t1, t2 = init_augmentations(args)

    trainset = USPTOConstrativeDataset(
        filename=args.trainset_filename,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        init_state_dict=state_dict_filename,
        num_processes=args.nprocs,
        transform_features=True,
        transform1=t1,
        transform2=t2,
    )
    state_dict = trainset.state_dict()

    valset = USPTOConstrativeDataset(
        filename=args.valset_filename,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
        transform_features=True,
        transform1=t1,
        transform2=t2,
    )

    testset = USPTOConstrativeDataset(
        filename=args.testset_filename,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
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

    # Add dataset state dict to args to log it
    args.dataset_state_dict = state_dict

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
    def select_transform(name, ratio, mask_value_atom, mask_value_bond):
        if name == "drop_atom":
            t = transforms.DropAtom(ratio=ratio)
        elif name == "drop_bond":
            t = transforms.DropBond(ratio=ratio)
        elif name == "mask_atom":
            t = transforms.MaskAtomAttribute(ratio=ratio, mask_value=mask_value_atom)
        elif name == "mask_bond":
            t = transforms.MaskBondAttribute(ratio=ratio, mask_value=mask_value_bond)
        else:
            raise ValueError(f"Unsupported augmentation type {name}")

        return t

    t1 = select_transform(
        args.augment_1,
        args.augment_1_ratio,
        args.augment_mask_value_atom,
        args.augment_mask_value_bond,
    )
    t2 = select_transform(
        args.augment_2,
        args.augment_2_ratio,
        args.augment_mask_value_atom,
        args.augment_mask_value_bond,
    )

    return t1, t2
