"""
Featurize atoms, bonds, and the global state of molecules with rdkit.
"""

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import GetPeriodicTable
from typing import List, Any, Optional


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


def bond_is_in_ring_one_hot(bond, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.IsInRing(), allowable_set, encode_unknown)


def bond_is_in_ring(bond):
    return [int(bond.IsInRing())]


def bond_in_ring_of_size_one_hot(bond_index, ring, allowable_set=None):
    if allowable_set is None:
        allowable_set = [3, 4, 5, 6, 7]
    return [int(ring.IsBondInRingOfSize(bond_index, s)) for s in allowable_set]


def bond_is_dative(bond):
    return [int(bond.GetBondType() == Chem.rdchem.BondType.DATIVE)]


def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)


def atom_degree(atom):
    return [atom.GetDegree()]


def atom_total_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(6))
    return one_hot_encoding(atom.GetTotalDegree(), allowable_set, encode_unknown)


def atom_total_degree(atom):
    return [atom.GetTotalDegree()]


def atom_is_in_ring_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.IsInRing(), allowable_set, encode_unknown)


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


def atom_type_one_hot(atom, allowable_set, encode_unknown=False):
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)


def global_num_atoms(mol):
    return [mol.GetNumAtoms()]


def global_num_bonds(mol):
    return [mol.GetNumBonds()]


def global_molecule_weight(mol):
    pt = GetPeriodicTable()
    return sum([pt.GetAtomicWeight(a.GetAtomicNum()) for a in mol.GetAtoms()])


def global_molecule_charge(mol):
    return [Chem.GetFormalCharge(mol)]


def global_molecule_charge_one_hot(mol, allowable_set, encode_unknown=False):
    return one_hot_encoding(Chem.GetFormalCharge(mol), allowable_set, encode_unknown)


class BaseFeaturizer:
    """
    Base featurizer for atoms, bonds, and the global state.
    """

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

    def __call__(self, mol) -> torch.Tensor:
        raise NotImplementedError


class BondFeaturizer(BaseFeaturizer):
    """
    Featurize bonds in a molecule.
    """

    @property
    def feature_name(self):

        mol = Chem.MolFromSmiles("CO")
        ring = mol.GetRingInfo()
        bond = mol.GetAtomWithIdx(0)

        names = (
            ["in_ring"] * len(bond_is_in_ring(bond))
            + ["in ring size"] * len(bond_in_ring_of_size_one_hot(0, ring))
            + ["dative"] * len(bond_is_dative(bond))
        )

        # Create a dummy molecule to make sure feature size is correct.
        # This is to avoid the case where we updated the __call__() function but forgot
        # the `feature_name()`
        mol = Chem.MolFromSmiles("CO")
        feat_size = self(mol).shape[1]
        if len(names) != feat_size:
            raise RuntimeError(
                f"Feature size calculated from feature name ({len(names)}) not equal to "
                f"that calculated from real feature({feat_size}). Probably you forget to "
                f"update the `feature_name()` function. "
            )

        return names

    def __call__(self, mol: Chem.Mol):
        """
        Args:
            mol: rdkit molecule

        Returns:
            A tensor of shape (N, D), where N is the number of features and D is the
            feature size. where N is the number of bonds in the molecule and D is the
            feature size. Note, when the molecule is a single atom molecule without bonds,
            a zero tensor of shape (1, D) is returned.
       """

        num_bonds = mol.GetNumBonds()
        if num_bonds == 0:
            feats = [[0.0 for _ in range(self.feature_size)]]
        else:
            ring = mol.GetRingInfo()
            feats = []
            for u in range(num_bonds):
                bond = mol.GetBondWithIdx(u)
                ft = bond_is_in_ring(bond)
                ft += bond_in_ring_of_size_one_hot(u, ring)
                ft += bond_is_dative(bond)
                feats.append(ft)

        feats = torch.tensor(feats, dtype=torch.float32)

        return feats


class AtomFeaturizer(BaseFeaturizer):
    """
    Featurize atoms in a molecule.
    """

    @property
    def feature_name(self):

        if not hasattr(self, "allowable_atom_type"):
            raise RuntimeError(
                "`feature_name` should be called after features are generated."
            )

        mol = Chem.MolFromSmiles("CO")
        ring = mol.GetRingInfo()
        atom = mol.GetAtomWithIdx(0)

        names = (
            ["atom type"]
            * len(atom_type_one_hot(atom, allowable_set=self.allowable_atom_type))
            + ["total degree"] * len(atom_total_degree_one_hot(atom))
            + ["is in ring"] * len(atom_is_in_ring(atom))
            + ["is in ring of size"] * len(atom_in_ring_of_size_one_hot(0, ring))
            + ["total num H"] * len(atom_total_num_H_one_hot(include_neighbors=True))
        )

        # Create a dummy molecule to make sure feature size is correct.
        # This is to avoid the case where we updated the __call__() function but forgot
        # the `feature_name()`
        feat_size = self(mol, self.allowed_atom_type).shape[1]
        if len(names) != feat_size:
            raise RuntimeError(
                f"Feature size calculated from feature name ({len(names)}) not equal to "
                f"that calculated from real feature({feat_size}). Probably you forget to "
                f"update the `feature_name()` function. "
            )

        return names

    def __call__(self, mol: Chem.Mol, allowable_atom_type: List[str]) -> torch.Tensor:
        """
        Args:
            mol: rdkit molecule
            allowable_atom_type: allowed atom species set for one-hot encoding

        Returns:
             2D tensor of shape (N, D), where N is the number of atoms, and D is the
             feature size.
        """

        # keep track of runtime argument
        self.allowed_atom_type = allowable_atom_type

        ring = mol.GetRingInfo()

        feats = []
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            ft = atom_type_one_hot(atom, allowable_set=allowable_atom_type)
            ft += atom_total_degree_one_hot(atom)
            ft += atom_is_in_ring(atom)
            ft += atom_in_ring_of_size_one_hot(i, ring)
            ft += atom_total_num_H_one_hot(include_neighbors=True)
            feats.append(ft)

        feats = torch.tensor(feats, dtype=torch.float32)

        return feats


class GlobalFeaturizer(BaseFeaturizer):
    """
    Featurize the global state of a molecules.


    Args:
        allowable_charge: allowed charges for molecule to take. If `None`, this feature
            is not used.
        allowable_solvent_environment: allowed solvent environment in which the
            calculations take place. If `None`, this feature is not used.
    """

    def __init__(self, allowable_charge=None, allowable_solvent_environment=None):
        super(GlobalFeaturizer, self).__init__()
        self.allowable_charge = allowable_charge
        self.allowable_solvent_environment = allowable_solvent_environment

    @property
    def feature_name(self):

        names = ["num atoms"] + ["num bonds"] + ["molecule weight"]
        if self.allowable_charge is not None:
            names += ["charge"] * len(self.allowable_charge)
        if self.allowable_solvent_environment is not None:
            if len(self.allowable_solvent_environment) == 2:
                names += ["environment"]
            else:
                names += ["environment"] * len(self.allowable_solvent_environment)

        # Create a dummy molecule to make sure feature size is correct.
        # This is to avoid the case where we updated the __call__() function but forgot
        # the `feature_name()`
        mol = Chem.MolFromSmiles("CO")
        if self.allowable_charge is not None:
            cg = self.allowable_charge[0]
        else:
            cg = None
        if self.allowable_solvent_environment is not None:
            env = self.allowable_solvent_environment[0]
        else:
            env = None
        feat_size = self(mol, charge=cg, environment=env).shape[1]

        if len(names) != feat_size:
            raise RuntimeError(
                f"Feature size calculated from feature name ({len(names)}) not equal to "
                f"that calculated from real feature({feat_size}). Probably you forget to "
                f"update the `feature_name()` function. "
            )

    def __call__(
        self, mol, charge: Optional[int] = None, environment: Optional[str] = None
    ):
        """
        Args:
            mol: rdkit molecule
            charge: charge of a molecule
            environment: solvent environment in which the computation is conducted

        Returns:
            2D tensor of shape (1, D), where D is the feature size.
        """

        ft = global_num_atoms() + global_num_bonds() + global_molecule_weight()

        # If `allowable_charge` is not `None` at instantiation, use this feature;
        # otherwise, ignore it.
        # If this feature is used, and `charge` is not `None`, the provided `charge`
        # will be used to for the feature; otherwise the formal charge of the molecule
        # is used.
        if self.allowable_charge is not None:
            if charge is not None:
                ft += one_hot_encoding(charge, self.allowable_charge)
            else:
                ft += global_molecule_charge_one_hot(mol, self.allowable_charge)

        # If `allowable_solvent_environment` is not `None` at instantiation, use this
        # feature; otherwise, ignore it.
        # If only two solvent_environment, we use 0/1 encoding. If more than two, we
        # use one-hot encoding.
        if self.allowable_solvent_environment is not None:
            if environment is None:
                raise ValueError(
                    "`allowable_solvent_environment` is not `None`. In this case, "
                    "expect `environment` not to be `None`, but got `None`."
                )
            if len(self.allowable_solvent_environment) == 2:
                ft += [self.allowable_solvent_environment.index(environment)]
            else:
                ft += one_hot_encoding(environment, self.allowable_solvent_environment)

        feats = [ft]
        feats = torch.tensor(feats, dtype=torch.float32)

        return feats
