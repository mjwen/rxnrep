"""
Featurize atoms, bonds, and the global state of molecules with rdkit.
"""

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdchem import GetPeriodicTable


def one_hot_encoding(x: Any, allowable_set: List[Any], encode_unknown: bool = False):
    """One-hot encoding.

    Args:
        x: Value to encode.
        allowable_set: The elements of the allowable_set should be of the same type as x.
        encode_unknown: If True, map inputs not in the allowable set to the additional
            last element.

    Returns:
        List of {0,1} values where at most one value is 1.
        The list is of length ``len(allowable_set)`` if ``encode_unknown=False``
        and ``len(allowable_set) + 1`` otherwise.
    """
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return [int(x == s) for s in allowable_set]


def bond_is_in_ring(bond):
    return [int(bond.IsInRing())]


def bond_in_ring_of_size_one_hot(bond_index, ring, allowable_set=None):
    if allowable_set is None:
        allowable_set = [3, 4, 5, 6, 7]
    return [int(ring.IsBondInRingOfSize(bond_index, s)) for s in allowable_set]


def bond_is_dative(bond):
    return [int(bond.GetBondType() == Chem.rdchem.BondType.DATIVE)]


def bond_is_conjugated(bond):
    return [int(bond.GetIsConjugated())]


def bond_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
            # Chem.rdchem.BondType.IONIC,
        ]
    return one_hot_encoding(atom.GetBondType(), allowable_set, encode_unknown)


def atom_type_one_hot(atom, allowable_set, encode_unknown=False):
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)


def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(6))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)


def atom_degree(atom):
    return [atom.GetDegree()]


def atom_total_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(6))
    return one_hot_encoding(atom.GetTotalDegree(), allowable_set, encode_unknown)


def atom_total_degree(atom):
    return [atom.GetTotalDegree()]


def atom_is_in_ring(atom):
    return [int(atom.IsInRing())]


def atom_in_ring_of_size_one_hot(atom_index, ring, allowable_set=None):
    if allowable_set is None:
        allowable_set = [3, 4, 5, 6, 7]
    return [int(ring.IsAtomInRingOfSize(atom_index, s)) for s in allowable_set]


def atom_total_num_H_one_hot(
    atom, allowable_set=None, encode_unknown=False, include_neighbors=False
):
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(
        atom.GetTotalNumHs(includeNeighbors=include_neighbors),
        allowable_set,
        encode_unknown,
    )


def atom_total_num_H(atom, include_neighbors=False):
    return [atom.GetTotalNumHs(includeNeighbors=include_neighbors)]


def atom_is_aromatic(atom):
    return [int(atom.GetIsAromatic())]


def atom_total_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(6))
    return one_hot_encoding(atom.GetTotalValence(), allowable_set, encode_unknown)


def atom_total_valence(atom):
    return [atom.GetTotalValence()]


def atom_num_radical_electrons_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(3))
    return one_hot_encoding(
        atom.GetNumRadicalElectrons(), allowable_set, encode_unknown
    )


def atom_num_radical_electrons(atom):
    return [atom.GetNumRadicalElectrons()]


def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            # Chem.rdchem.HybridizationType.SP3D,
            # Chem.rdchem.HybridizationType.SP3D2,
        ]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)


def atom_formal_charge(atom):
    return [atom.GetFormalCharge()]


def atom_formal_charge_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    return one_hot_encoding(atom.GetFormalCharge(), allowable_set, encode_unknown)


def global_num_atoms(mol):
    return [mol.GetNumAtoms()]


def global_num_bonds(mol):
    return [mol.GetNumBonds()]


def global_molecule_weight(mol):
    pt = GetPeriodicTable()
    return [sum([pt.GetAtomicWeight(a.GetAtomicNum()) for a in mol.GetAtoms()])]


def global_molecule_charge(mol, charge):
    if charge is None:
        charge = Chem.GetFormalCharge(mol)
    return [charge]


def global_molecule_charge_one_hot(mol, charge, allowable_set, encode_unknown=False):
    if charge is None:
        charge = Chem.GetFormalCharge(mol)
    return one_hot_encoding(charge, allowable_set, encode_unknown)


def global_molecule_spin(spin):
    return [spin]


def global_molecule_spin_one_hot(spin, allowable_set, encode_unknown=False):
    return one_hot_encoding(spin, allowable_set, encode_unknown)


def global_molecule_environment(environment, allowable_set, encode_unknown=False):
    if len(allowable_set) == 2:
        return [allowable_set.index(environment)]
    else:
        return one_hot_encoding(environment, allowable_set, encode_unknown)


class BaseFeaturizer:
    """
    Base featurizer for atoms, bonds, and the global state.

    Args:
        featurizers: functions used for featurization. Could also be string name of the
            functions.
        featurizer_kwargs: extra keyword arguments for the featurizers, e.g. used to
            provide allowable set for the function. A dictionary {fn_name, fn_kwargs},
            where `fn_name` is a string of the function name, and `fn_kwargs` is a
            dictionary {kwargs_name, kwargs_value} of the keyword arguments.
            Default to None to use preset default values.
    """

    DEFAULTS = []

    def __init__(
        self,
        featurizers: Optional[List[Union[str, Callable]]] = None,
        featurizer_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):

        if featurizers is None:
            featurizers = self.DEFAULTS
        if featurizer_kwargs is None:
            featurizer_kwargs = {}

        self.featurizers = OrderedDict()  # {fn_name: {"fn":fn, "kwargs": kwargs}}
        for f in featurizers:

            if isinstance(f, str):
                # provided string name of functions
                name = f
                try:
                    fn = globals()[name]
                except ValueError:
                    raise RuntimeError(f"Cannot find featurizer function {name}")
            else:
                # directly provide function object
                fn = f
                name = f.__name__

            kwargs = featurizer_kwargs.get(name, {})

            self.featurizers[name] = {"fn": fn, "kwargs": kwargs}

        # check again that every provided featurizer_kwargs is used
        for name in featurizer_kwargs:
            if name not in self.featurizers:
                raise ValueError(
                    f"Provided featurizer kwargs {name} in `featurizer_kwargs` does not "
                    f"have a corresponding featurizing function."
                )

    @property
    def feature_name(self) -> List[str]:
        """
        Returns a list of the names of each feature. Should be of the same length as
        `feature_size`.
        """
        raise NotImplementedError

    @property
    def feature_size(self) -> int:
        """
        Returns size of the features.
        """
        return len(self.feature_name)

    def __call__(
        self, mol: Chem.Mol, mol_property: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        raise NotImplementedError


class BondFeaturizer(BaseFeaturizer):
    """
    Featurize bonds in a molecule.
    """

    DEFAULTS = [
        bond_is_in_ring,
        bond_in_ring_of_size_one_hot,
        bond_is_conjugated,
        bond_type_one_hot,
    ]

    @property
    def feature_name(self) -> List[str]:

        mol = Chem.MolFromSmiles("CO")
        ring = mol.GetRingInfo()
        index = 0
        bond = mol.GetBondWithIdx(index)

        index1 = 0
        index2 = 1

        mol_property = _get_fake_mol_property()

        # feats of a bond
        ft_names = []
        for name, value in self.featurizers.items():
            fn = value["fn"]
            kwargs = value["kwargs"]

            if name == "bond_in_ring_of_size_one_hot":
                ft = fn(index, ring, **kwargs)
            elif name == "bond_distance":
                ft = fn(index1, index2, mol_property)
            else:
                ft = fn(bond, **kwargs)
            ft_names += [name] * len(ft)

        # Create a dummy molecule to make sure feature size is correct.
        # This is to avoid the case where we updated the __call__() function but forgot
        # the `feature_name()`
        mol = Chem.MolFromSmiles("CO")
        feat_size = self(mol, mol_property=mol_property).shape[1]
        if len(ft_names) != feat_size:
            raise RuntimeError(
                f"Feature size calculated from feature name ({len(ft_names)}) not equal "
                f"to that calculated from real feature({feat_size}). Probably you forget "
                f"to update the `feature_name()` function."
            )

        return ft_names

    def __call__(
        self,
        mol: Chem.Mol,
        mol_property: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Args:
            mol: rdkit molecule
            mol_property: additional property of molecules, e.g. partial charge,
                partial spin and coords. If None, not used.

        Returns:
            A tensor of shape (N, D), where N is the number of bonds in the molecule
            and D is the feature size. Note, when the molecule is a single atom molecule
            without bonds, a zero tensor of shape (1, D) is returned.
        """

        num_bonds = mol.GetNumBonds()
        if num_bonds == 0:
            feats = torch.tensor([], dtype=torch.float32).reshape(0, self.feature_size)
        else:
            ring = mol.GetRingInfo()

            # feats of all bonds
            feats = []
            for index in range(num_bonds):
                bond = mol.GetBondWithIdx(index)
                index1 = bond.GetBeginAtomIdx()
                index2 = bond.GetEndAtomIdx()

                # feats of a bond
                ft = []
                for name, value in self.featurizers.items():
                    fn = value["fn"]
                    kwargs = value["kwargs"]

                    if name == "bond_in_ring_of_size_one_hot":
                        ft += fn(index, ring, **kwargs)
                    elif name == "bond_distance":
                        assert isinstance(mol_property, dict), (
                            f"Featurizer function `{name}` requires `mol_property` "
                            "provided as dictionary"
                        )
                        ft += fn(index1, index2, mol_property)
                    else:
                        ft += fn(bond, **kwargs)

                feats.append(ft)

            feats = torch.tensor(feats, dtype=torch.float32)

        return feats


class AtomFeaturizer(BaseFeaturizer):
    """
    Featurize atoms in a molecule.
    """

    DEFAULTS = [
        atom_type_one_hot,
        atom_total_degree_one_hot,
        atom_is_in_ring,
        atom_in_ring_of_size_one_hot,
        atom_total_num_H_one_hot,
        atom_is_aromatic,
        atom_total_valence_one_hot,
        atom_num_radical_electrons_one_hot,
        atom_hybridization_one_hot,
    ]

    @property
    def feature_name(self):

        if not hasattr(self, "allowable_atom_type"):
            raise RuntimeError(
                "`feature_name` should be called after features are generated."
            )

        mol = Chem.MolFromSmiles("CO")
        ring = mol.GetRingInfo()
        index = 0
        atom = mol.GetAtomWithIdx(index)

        mol_property = _get_fake_mol_property()

        # feats of an atom
        ft_names = []
        for name, value in self.featurizers.items():
            fn = value["fn"]
            kwargs = value["kwargs"]

            if name == "atom_type_one_hot":
                ft = fn(atom, allowable_set=self.allowable_atom_type, **kwargs)
            elif name == "atom_in_ring_of_size_one_hot":
                ft = fn(index, ring, **kwargs)
            elif name in ["atom_total_num_H_one_hot", "atom_total_num_H"]:
                ft = fn(atom, include_neighbors=True, **kwargs)
            elif name in [
                "atom_resp_partial_charge",
                "atom_mulliken_partial_charge",
                "atom_critic2_partial_charge",
                "atom_mulliken_partial_spin",
            ]:
                ft = fn(index, mol_property)
            else:
                ft = fn(atom, **kwargs)

            ft_names += [name] * len(ft)

        # Create a dummy molecule to make sure feature size is correct.
        # This is to avoid the case where we updated the __call__() function but forgot
        # the `feature_name()`
        mol = Chem.MolFromSmiles("CO")
        feat_size = self(
            mol, self.allowable_atom_type, mol_property=mol_property
        ).shape[1]
        if len(ft_names) != feat_size:
            raise RuntimeError(
                f"Feature size calculated from feature name ({len(ft_names)}) not equal "
                f"to that calculated from real feature({feat_size}). Probably you forget "
                f"to update the `feature_name()` function. "
            )

        return ft_names

    def __call__(
        self,
        mol: Chem.Mol,
        allowable_atom_type: List[str],
        mol_property: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Args:
            mol: rdkit molecule
            allowable_atom_type: allowed atom species set for one-hot encoding
            mol_property: additional property of molecules, e.g. partial charge,
                partial spin and coords. If None, not used.

        Returns:
             2D tensor of shape (N, D), where N is the number of atoms, and D is the
             feature size.
        """

        # keep track of runtime argument
        self.allowable_atom_type = allowable_atom_type

        ring = mol.GetRingInfo()

        # feats of all atoms
        feats = []
        for index in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(index)

            # feats of an atom
            ft = []
            for name, value in self.featurizers.items():
                fn = value["fn"]
                kwargs = value["kwargs"]

                if name == "atom_type_one_hot":
                    ft += fn(atom, allowable_set=self.allowable_atom_type, **kwargs)
                elif name == "atom_in_ring_of_size_one_hot":
                    ft += fn(index, ring, **kwargs)
                elif name in ["atom_total_num_H_one_hot", "atom_total_num_H"]:
                    ft += fn(atom, include_neighbors=True, **kwargs)
                elif name in [
                    "atom_resp_partial_charge",
                    "atom_mulliken_partial_charge",
                    "atom_critic2_partial_charge",
                    "atom_mulliken_partial_spin",
                ]:

                    assert isinstance(mol_property, dict), (
                        f"Featurizer function `{name}` requires `mol_property` "
                        "provided as dictionary"
                    )
                    ft += fn(index, mol_property)
                else:
                    ft += fn(atom, **kwargs)

            feats.append(ft)

        feats = torch.tensor(feats, dtype=torch.float32)

        return feats


class GlobalFeaturizer(BaseFeaturizer):
    """
    Featurize the global state of a molecules.

    Args:
        featurizers: functions used for featurization. Could also be string name of the
            functions.
        featurizer_kwargs: extra keyword arguments for the featurizers, e.g. used to
            provide allowable set for the function. A dictionary {fn_name, fn_kwargs},
            where `fn_name` is a string of the function name, and `fn_kwargs` is a
            dictionary {kwargs_name, kwargs_value} of the keyword arguments.
            Default to None to use preset default values.
        allowable_charge: allowed charges for molecule to take. If `None`, this feature
            is not used.
        allowable_spin: allowed spins for molecule to take. If `None`, this feature
            is not used.
        allowable_solvent_environment: allowed solvent environment in which the
            calculations take place. If `None`, this feature is not used.
    """

    DEFAULTS = [global_num_atoms, global_num_bonds, global_molecule_weight]

    def __init__(
        self,
        featurizers: Optional[List[Union[str, Callable]]] = None,
        featurizer_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        allowable_charge: List[int] = None,
        allowable_spin: List[int] = None,
        allowable_solvent_environment: List[str] = None,
    ):
        super().__init__(featurizers, featurizer_kwargs)

        # add charge, spin and environment featurizer
        if allowable_charge is not None:
            fn = global_molecule_charge_one_hot
            name = fn.__name__
            self.featurizers[name] = {
                "fn": fn,
                "kwargs": {"allowable_set": allowable_charge},
            }

        if allowable_spin is not None:
            fn = global_molecule_spin_one_hot
            name = fn.__name__
            self.featurizers[name] = {
                "fn": fn,
                "kwargs": {"allowable_set": allowable_spin},
            }

        if allowable_solvent_environment is not None:
            fn = global_molecule_environment
            name = fn.__name__
            self.featurizers[name] = {
                "fn": fn,
                "kwargs": {"allowable_set": allowable_solvent_environment},
            }

    @property
    def feature_name(self):

        mol = Chem.MolFromSmiles("CO")
        charge = None
        spin = None
        environment = None

        ft_names = []
        for name, value in self.featurizers.items():
            fn = value["fn"]
            kwargs = value["kwargs"]

            if name == "global_molecule_charge_one_hot":
                charge = kwargs["allowable_set"][0]
                ft = fn(mol, charge, **kwargs)
            elif name == "global_molecule_spin_one_hot":
                spin = kwargs["allowable_set"][0]
                ft = fn(spin, **kwargs)
            elif name == "global_molecule_environment":
                environment = kwargs["allowable_set"][0]
                ft = fn(environment, **kwargs)
            else:
                ft = fn(mol, **kwargs)

            ft_names += [name] * len(ft)

        # Create a dummy molecule to make sure feature size is correct.
        # This is to avoid the case where we updated the __call__() function but forgot
        # the `feature_name()`
        mol = Chem.MolFromSmiles("CO")
        feat_size = self(mol, charge=charge, spin=spin, environment=environment).shape[
            1
        ]

        if len(ft_names) != feat_size:
            raise RuntimeError(
                f"Feature size calculated from feature name ({len(ft_names)}) not equal "
                f"to that calculated from real feature({feat_size}). Probably you "
                f"forget to update the `feature_name()` function."
            )

        return ft_names

    def __call__(
        self,
        mol,
        charge: Optional[int] = None,
        spin: Optional[int] = None,
        environment: Optional[str] = None,
    ):
        """
        Args:
            mol: rdkit molecule
            charge: charge of a molecule. The behavior of the charge feature depends on
                `allowable_charge`. If `allowable_charge=None`, this feature is not
                used. When `allowable_charge` is a list of charges, the `charge` will
                be a one hot feature of `allowable_charge`. If `charge=None`,
                the formal charge of the rdkit molecule will be used.
            spin: spin of a molecule.
            environment: solvent environment in which the computation is conducted

        Returns:
            2D tensor of shape (1, D), where D is the feature size.
        """

        ft = []
        for name, value in self.featurizers.items():
            fn = value["fn"]
            kwargs = value["kwargs"]

            if name == "global_molecule_charge_one_hot":
                ft += fn(mol, charge, **kwargs)
            elif name == "global_molecule_spin_one_hot":
                ft += fn(spin, **kwargs)
            elif name == "global_molecule_environment":
                ft += fn(environment, **kwargs)
            else:
                ft += fn(mol, **kwargs)

        feats = [ft]
        feats = torch.tensor(feats, dtype=torch.float32)

        return feats


class MoleculeFeaturizer:
    def feature_name(self):
        raise NotImplementedError

    def feature_size(self):
        raise NotImplementedError

    def __call__(self, mol: Chem.Mol) -> torch.Tensor:
        raise NotImplementedError


class MorganFeaturizer(MoleculeFeaturizer):
    def __init__(self, radius: int = 2, size: int = 2048):
        self.radius = radius
        self.size = size

    def feature_name(self):
        return "Morgan features"

    def feature_size(self):
        return self.size

    def __call__(self, mol: Chem.Mol) -> torch.Tensor:
        feats = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.radius,
            nBits=self.size,
            # useChirality=self.chiral,
            # useBondTypes=self.bonds,
            # useFeatures=self.features,
        )
        feats = torch.from_numpy(np.asarray(feats, dtype=np.float32))

        return feats


def _get_fake_mol_property():
    """
    Fake mol property for CO used for determining feature name.
    """
    return {
        "resp_partial_charge": [0.1, -0.1],
        "mulliken_partial_charge": [0.1, -0.1],
        "critic2_partial_charge": [0.1, -0.1],
        "mulliken_partial_spin": [0.1, -0.1],
        "coords": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }
