import torch

from rxnrep.core.molecule import Molecule
from rxnrep.data.to_graph import mol_to_graph


def create_graph(m, n_a, n_b, n_v):
    """

    Args:
        m: molecule
        n_a:  number of atoms
        n_b:  number of bonds
        n_v:  number of global nodes
    """

    g = mol_to_graph(m, num_global_nodes=n_v)

    g.nodes["atom"].data.update({"feat": torch.arange(n_a * 4).float().reshape(n_a, 4)})

    # this create zero tensor (i.e. of shape (0, 3)) if n_b == 0
    bond_feats = torch.arange(n_b * 3).float().reshape(n_b, 3)
    bond_feats = torch.repeat_interleave(bond_feats, 2, dim=0)
    g.edges["bond"].data.update({"feat": bond_feats})

    if n_v > 0:
        g.nodes["global"].data.update(
            {"feat": torch.arange(n_v * 2).float().reshape(n_v, 2)}
        )

    return g


def create_graph_C(num_global_nodes):
    """
    Create a single atom molecule C.

    atom_feats:
        [[0,1,2,3]]

    bond_feats:
        None

    global_feats:
        [[0,1],
         [2,3],
         ...]
    """
    smi = "[C]"
    m = Molecule.from_smiles(smi)

    return create_graph(m, 1, 0, num_global_nodes)


def create_graph_CO2(num_global_nodes):
    """
    Create a CO2 and add features.

    Molecule:
          0        1
    O(0) --- C(1) --- O(2)

    atom_feat:
        [[0,1,2,3],
         [4,5,6,7],
         [8,9,10,11]]

    bond_feat:
        [[0,1,2],
         [0,1,2],
         [3,4,5],
         [3,4,5]]
    Note [0,1,2] corresponds to two edges for the same bond.


    global_feat:
        [[0,1],
         [2,3],
         ... ]

    """
    smi = "O=C=O"
    m = Molecule.from_smiles(smi)

    return create_graph(m, 3, 2, num_global_nodes)
