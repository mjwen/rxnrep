from __future__ import annotations

import copy
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)


class Molecule:
    """
    A wrapper over rdkit molecule to make it easier to use.

    Args:
        mol: rdkit molecule.
        id: an identification of the molecule.
    """

    def __init__(self, mol: Chem.Mol, id: Optional[Union[int, str]] = None):

        self._mol = mol
        self._id = id

        self._charge = None
        self._environment = None

    @classmethod
    def from_smiles(cls, s: str, sanitize: bool = True):
        """
        Create a molecule from a smiles string.

        Args:
            s: smiles string, e.g. [CH3+]
            sanitize: whether to sanitize the molecule
        """
        m = Chem.MolFromSmiles(s, sanitize=sanitize)
        if m is None:
            raise MoleculeError(f"Cannot create molecule for: {s}")
        return cls(m, s)

    @classmethod
    def from_smarts(cls, s: str, sanitize: bool = True):
        """
        Create a molecule for a smarts string.

        Args:
            s: smarts string, e.g. [Cl:1][CH2:2][CH2:3][CH2:4][C:5](Cl)=[O:6]
            sanitize: whether to sanitize the molecule.
        """
        m = Chem.MolFromSmarts(s)
        if m is None:
            raise MoleculeError(f"Cannot create molecule for: {s}")
        if sanitize:
            Chem.SanitizeMol(m)
        return cls(m, s)

    @classmethod
    def from_sdf(cls, s: str, sanitize: bool = True, remove_H: bool = False):
        """
        Create a molecule for a sdf molecule block.

        We choose to set the default of `remove_H` to `False` because SDF definition of
        explicit and implicit hydrogens is a bit different from what in smiles: it is
        not true that hydrogens specified in SDF are explicit; whether a
        hydrogen is explict or implicit depends on the charge(CHG), valence(VAL) and
        radicals(RAD) specified in the SDF file.

        Args:
            s: SDF mol block string. .
            sanitize: whether to sanitize the molecule
            remove_H: whether to remove hydrogens read from SDF
        """
        m = Chem.MolFromMolBlock(s, sanitize=sanitize, removeHs=remove_H)
        if m is None:
            raise MoleculeError(f"Cannot create molecule for: {s}")
        return cls(m)

    @property
    def rdkit_mol(self) -> Chem.Mol:
        """
        Returns the underlying rdkit molecule..
        """
        return self._mol

    @property
    def id(self) -> Union[int, str, None]:
        """
        Returns the identifier of the molecule.
        """
        return self._id

    @property
    def formal_charge(self) -> int:
        """
        Returns formal charge of the molecule.
        """
        return Chem.GetFormalCharge(self._mol)

    @property
    def charge(self) -> int:
        """
        Returns charge of the molecule.

        The returned charge is the `formal_charge` of the underlying rdkit molecule,
        if charge is set. Otherwise, it is the set charge, which could be different
        from the formal charge.
        """
        if self._charge is None:
            return self.formal_charge
        else:
            return self._charge

    @charge.setter
    def charge(self, charge: int):
        """
        Set the charge of a molecule.

        This will not alter the underlying rdkit molecule at all.

        The charge could be different from the formal charge of the molecule. The purpose
        is to host a charge value that could be used when fragmenting molecules and so on.

        Args:
            charge: charge of the molecule
        """
        self._charge = charge

    @property
    def num_atoms(self) -> int:
        """
        Returns number of atoms in molecule
        """
        return self._mol.GetNumAtoms()

    @property
    def num_bonds(self) -> int:
        """
        Returns number of bonds in the molecule.
        """
        return self._mol.GetNumBonds()

    @property
    def bonds(self) -> List[Tuple[int, int]]:
        """
        Returns bond indices specified as tuples of atom indices.
        """
        indices = [
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in self._mol.GetBonds()
        ]

        return indices

    @property
    def species(self) -> List[str]:
        """
        Return species of atoms.
        """
        return [a.GetSymbol() for a in self._mol.GetAtoms()]

    @property
    def composition_dict(self) -> Dict[str, int]:
        """
        Returns composition of the molecule with species as key and number of the species
        as value.
        """
        comp = defaultdict(int)
        for s in self.species:
            comp[s] += 1

        return comp

    @property
    def formula(self) -> str:
        """
        Returns chemical formula of the molecule, e.g. C1H2O3.
        """
        comp = self.composition_dict
        f = ""
        for s in sorted(set(self.species)):
            f += f"{s}{comp[s]}"

        return f

    @property
    def coords(self) -> np.ndarray:
        """
        Returns coordinates of the atoms. The 2D array is of shape (N, 3), where N is the
        number of atoms.

        Conformer needs to be created before calling this. Conformer can be created by
        embedding the atom.
        """
        # coords = self._mol.GetConformer().GetPositions()
        # NOTE, the above way to get coords results in segfault on linux, so we use the
        # below workaround
        coords = [
            [float(x) for x in self._mol.GetConformer().GetAtomPosition(i)]
            for i in range(self._mol.GetNumAtoms())
        ]

        return np.asarray(coords)

    @property
    def environment(self) -> str:
        """
        Return the computation environment of the molecule, e.g. solvent model.
        """
        return self._environment

    @environment.setter
    def environment(self, value: str):
        """
        Set the computation environment of the molecule, e.g. solvent model.
        """
        self._environment = value

    def get_atom_map_number(self) -> List[Union[int, None]]:
        """
        Get the atom map number in the rdkit molecule.

        Returns:
            Atom map number for each atom. Index in the returned list is the atom index.
            If an atom is not mapped, the map number is set to `None`.
        """

        map_number = []
        for i, atom in enumerate(self._mol.GetAtoms()):
            if atom.HasProp("molAtomMapNumber"):
                map_number.append(atom.GetAtomMapNum())
            else:
                map_number.append(None)

        return map_number

    def set_atom_map_number(self, map_number: Dict[int, int]):
        """
        Set the atom map number of the rdkit molecule.

        Args:
            Atom map number for each atom. If a value is `None`, the atom map number
            in the rdkit molecule is cleared.
        """
        for idx, number in map_number.items():
            if idx >= self.num_atoms:
                raise MoleculeError(
                    f"Cannot set atom map number of atom {idx} (starting from 0) for "
                    f"a molecule has a total number of {self.num_atoms} atoms."
                )

            atom = self._mol.GetAtomWithIdx(idx)

            if number is None:
                atom.ClearProp("molAtomMapNumber")
            elif number <= 0:
                raise MoleculeError(
                    f"Expect atom map number larger than 0, but got  {number}."
                )
            else:
                atom.SetAtomMapNum(number)

    def generate_coords(self) -> np.ndarray:
        """
        Generate 3D coordinates for an rdkit molecule by embedding it.

        The returned coordinates is a 2D array of shape (N, 3), where N is the number of
        atoms.
        """
        error = AllChem.EmbedMolecule(self._mol, randomSeed=35)
        if error == -1:  # https://sourceforge.net/p/rdkit/mailman/message/33386856/
            AllChem.EmbedMolecule(self._mol, randomSeed=35, useRandomCoords=True)
        if error == -1:
            raise MoleculeError(
                "Cannot generate coordinates for molecule; embedding fails."
            )

        return self.coords

    def optimize_coords(self) -> np.ndarray:
        """
        Optimize atom coordinates using MMFF and UFF force fields.

        Returns:
            optimized coords, a 2D array of shape (N, 3), where N is the number of atoms.
        """

        # TODO usually, you need to add the H
        def optimize_till_converge(method, m):
            maxiters = 200
            while True:
                error = method(m, maxIters=maxiters)
                if error == 1:
                    maxiters *= 2
                else:
                    return error

        # generate conformer if not exists
        try:
            self._mol.GetConformer()
        except ValueError:
            self.generate_coords()

        # optimize, try MMFF first, if fails then UFF
        error = optimize_till_converge(AllChem.MMFFOptimizeMolecule, self._mol)
        if error == -1:  # MMFF cannot be set up
            optimize_till_converge(AllChem.UFFOptimizeMolecule, self._mol)

        return self.coords

    def to_smiles(self) -> str:
        """
        Returns a smiles representation of the molecule.
        """
        return Chem.MolToSmiles(self._mol)

    def to_sdf(
        self,
        filename: Optional[Union[Path]] = None,
        kekulize: bool = True,
        v3000: bool = True,
        name: Optional[str] = None,
    ) -> str:
        """
        Convert molecule to an sdf representation.

        Args:
            filename: if not None, write to the path.
            kekulize: whether to kekulize the mol
            v3000: if `True` write in SDF v3000 format, otherwise, v2000 format.
            name: Name of the molecule, i.e. first line of the sdf file. If None,
            molecule.id will be used.

        Returns:
             a sdf representation of the molecule.
        """
        name = str(self._id) if name is None else name
        self._mol.SetProp("_Name", name)

        sdf = Chem.MolToMolBlock(self._mol, kekulize=kekulize, forceV3000=v3000)
        if filename is not None:
            with open(filename, "w") as f:
                f.write(sdf)

        return sdf

    def draw(
        self, filename: Optional[Union[Path]] = None, with_atom_index: bool = False
    ) -> Chem.Mol:
        """
        Draw the molecule.

        Args:
            filename: path to the save the generated image. If `None`,
                image will not be generated.
            with_atom_index: whether to show the atom index in the image.

        Returns:
            the molecule (which will then show up in Jupyter notebook)
        """
        # compute better coords to show in 2D
        m = copy.deepcopy(self._mol)
        AllChem.Compute2DCoords(m)

        if with_atom_index:
            for a in m.GetAtoms():
                a.SetAtomMapNum(a.GetIdx() + 1)

        if filename is not None:
            Draw.MolToFile(m, str(filename))

        return m

    def draw_with_bond_note(
        self,
        note: Dict[Tuple[int, int], str],
        filename: Optional[Union[Path]] = None,
        with_atom_index: bool = False,
    ):
        """
        Draw molecule and show a note along bond.

        Args:
            note: {bond_index: note}. The note to show for the corresponding bond.
                `bond index` is a tuple of atom indices.
            filename: path to the save the generated image. If `None`,
                image will not be generated, but instead, will show in Jupyter notebook.
            with_atom_index: whether to show the atom index in the image.
        """
        m = self.draw(with_atom_index=with_atom_index)

        # set bond annotation
        highlight_bonds = []
        for bond, note in note.items():
            if isinstance(note, (float, np.floating)):
                note = "{:.3g}".format(note)
            idx = m.GetBondBetweenAtoms(*bond).GetIdx()
            m.GetBondWithIdx(idx).SetProp("bondNote", note)
            highlight_bonds.append(idx)

        # set highlight color
        bond_colors = {b: (192 / 255, 192 / 255, 192 / 255) for b in highlight_bonds}

        d = rdMolDraw2D.MolDraw2DCairo(400, 300)

        # smaller font size
        d.SetFontSize(0.8 * d.FontSize())

        rdMolDraw2D.PrepareAndDrawMolecule(
            d, m, highlightBonds=highlight_bonds, highlightBondColors=bond_colors
        )
        d.FinishDrawing()

        if filename is not None:
            with open(filename, "wb") as f:
                f.write(d.GetDrawingText())

        # TODO the returned d may not show in Jupyter notebooks
        #  write to /tmp or using tempfile, and then display it using Ipython.display
        #  Also, check whether it is jupyter kernel

    def add_H(self, explicit_only: bool = False) -> Molecule:
        """
        Add hydrogens to the molecule.

        Args:
            explicit_only: only add explicit hydrogens to the graph

        Returns:
            The molecule with hydrogens added.
        """
        self._mol = Chem.AddHs(self._mol, explicitOnly=explicit_only)

        return self

    def remove_H(self, implicit_only: bool = False) -> Molecule:
        """
        Remove hydrogens to the molecule.

        Args:
            implicit_only: only remove implicit hydrogens from the graph

        Returns:
            The molecule with hydrogens removed.
        """
        self._mol = Chem.RemoveHs(self._mol, implicitOnly=not implicit_only)

        return self


class MoleculeError(Exception):
    def __init__(self, msg=None):
        super(MoleculeError, self).__init__(msg)
        self.msg = msg
