import itertools

import dgl
import torch

from rxnrep.core.reaction import smiles_to_reaction
from rxnrep.data.to_graph import combine_graphs, create_reaction_graph
from rxnrep.data.transforms import (
    DropAtom,
    DropBond,
    MaskAtomAttribute,
    MaskBondAttribute,
    Subgraph,
    SubgraphBFS,
    get_node_subgraph,
)
from rxnrep.utils import seed_all

from ..utils import create_graph


def test_drop_atom():
    seed_all(25)

    reactants_g, products_g, reaction_g, reaction = create_reaction_and_graphs()

    transform = DropAtom(ratio=0.5)
    sub_reactants_g, sub_products_g, sub_reaction_g, _ = transform(
        reactants_g, products_g, reaction_g, reaction
    )

    # atom 3 is dropped
    idx = 3

    select = [True for _ in range(6)]
    select[idx] = False

    # bond 1, and 4 are kept
    retained_bond_edges = [2, 3, 8, 9]

    assert_atom_subgroup(reactants_g, sub_reactants_g, select, retained_bond_edges)
    assert_atom_subgroup(products_g, sub_products_g, select, retained_bond_edges)


def test_drop_bond():
    seed_all(25)

    reactants_g, products_g, reaction_g, reaction = create_reaction_and_graphs()

    transform = DropBond(ratio=0.3)
    sub_reactants_g, sub_products_g, sub_reaction_g, _ = transform(
        reactants_g, products_g, reaction_g, reaction
    )

    # bond 1 is dropped
    idx = 1
    retained_bond_edges = list(
        itertools.chain.from_iterable(
            [[2 * i, 2 * i + 1] for i in range(5) if i != idx]
        )
    )

    assert_bond_subgroup(reactants_g, sub_reactants_g, retained_bond_edges)
    assert_bond_subgroup(products_g, sub_products_g, retained_bond_edges)


def test_mask_atom_attribute():
    seed_all(25)

    reactants_g, products_g, reaction_g, reaction = create_reaction_and_graphs()
    transform = MaskAtomAttribute(ratio=0.5, mask_value=1)
    sub_reactants_g, sub_products_g, sub_reaction_g, _ = transform(
        reactants_g, products_g, reaction_g, reaction
    )

    # atom 3 is selected
    idx = 3
    ref = torch.ones(3)
    assert torch.equal(sub_reactants_g.nodes["atom"].data["feat"][idx], ref)
    assert torch.equal(sub_products_g.nodes["atom"].data["feat"][idx], ref)


def test_mask_bond_attribute():

    seed_all(25)

    reactants_g, products_g, reaction_g, reaction = create_reaction_and_graphs()
    transform = MaskBondAttribute(ratio=0.3, mask_value=1)
    sub_reactants_g, sub_products_g, sub_reaction_g, _ = transform(
        reactants_g, products_g, reaction_g, reaction
    )

    # bond 1 is selected
    idx = 1
    indices = [2 * idx, 2 * idx + 1]  # each bond has two edges
    ref = torch.ones(2, 3)
    assert torch.equal(sub_reactants_g.edges["bond"].data["feat"][indices], ref)
    assert torch.equal(sub_products_g.edges["bond"].data["feat"][indices], ref)


def test_subgraph():

    reactants_g, products_g, reaction_g, reaction = create_reaction_and_graphs()
    transform = Subgraph(ratio=0.5)
    sub_reactants_g, sub_products_g, sub_reaction_g, _ = transform(
        reactants_g, products_g, reaction_g, reaction
    )

    # atom 3 is selected, atoms 0,2,4 are in center
    select = [True, False, True, True, True, False]

    # bonds 2, 4 are kept
    retained_bond_edges = [4, 5, 8, 9]

    assert_atom_subgroup(reactants_g, sub_reactants_g, select, retained_bond_edges)
    assert_atom_subgroup(products_g, sub_products_g, select, retained_bond_edges)


def test_subgraph_bfs():
    reactants_g, products_g, reaction_g, reaction = create_reaction_and_graphs()
    transform = SubgraphBFS(ratio=0.5)
    sub_reactants_g, sub_products_g, sub_reaction_g, _ = transform(
        reactants_g, products_g, reaction_g, reaction
    )

    # atom 3 is selected, atoms 0,2,4 are in center
    select = [True, False, True, True, True, False]

    # bonds 2, 4 are kept
    retained_bond_edges = [4, 5, 8, 9]

    assert_atom_subgroup(reactants_g, sub_reactants_g, select, retained_bond_edges)
    assert_atom_subgroup(products_g, sub_products_g, select, retained_bond_edges)


def test_node_subgraph():
    reactants_g, products_g, reaction_g, reaction = create_reaction_and_graphs()

    # after removing atom 3, bond 1, and 4 are kept
    retained_nodes = [0, 1, 2, 4, 5]
    retained_bond_edges = [2, 3, 8, 9]

    sub_g = get_node_subgraph(reactants_g, nodes=retained_nodes)
    assert_atom_subgroup(reactants_g, sub_g, retained_nodes, retained_bond_edges)

    sub_g = get_node_subgraph(products_g, nodes=retained_nodes)
    assert_atom_subgroup(products_g, sub_g, retained_nodes, retained_bond_edges)


def create_reaction_and_graphs():
    r"""Create a reaction: CH3CH2+ + CH3CH2CH2 --> CH3 + CH3CH2CH2CH2+

    m1:
        4
    C2-----C0

    m2:
        0      2
    C1-----C3-----C4
     \     |
     1 \   |  3
         \ C5

    m3:

    C2

    m4:
        0      2      4
    C1-----C3-----C4-----C0
     \     |
     1 \   |  3
         \ C5


    combined union graph (the bond index is assigned to mimic combine_graphs() function)

              4
          C2-----C0
                  |
                  |  5
        0      2  |
    C1-----C3-----C4
     \     |
     1 \   |  3
         \ C5

    """
    smi_rxn = "[CH3:3][CH2+:1].[CH2:2]1[CH1:4]([CH2:6]1)[CH2:5]>>[CH3:3].[CH2:2]1[CH1:4]([CH2:6]1)[CH2:5][CH2+:1]"
    reaction = smiles_to_reaction(smi_rxn)
    reactants = reaction.reactants
    products = reaction.products

    reactant_graphs = [
        create_graph(reactants[0], 2, 1, 1),
        create_graph(reactants[1], 4, 4, 1),
    ]
    product_graphs = [
        create_graph(products[0], 1, 0, 1),
        create_graph(products[1], 5, 5, 1),
    ]

    # combine small graphs to form one big graph for reactants and products
    atom_map_number = reaction.get_reactants_atom_map_number(zero_based=True)
    bond_map_number = reaction.get_reactants_bond_map_number(for_changed=True)
    reactants_g = combine_graphs(reactant_graphs, atom_map_number, bond_map_number)

    atom_map_number = reaction.get_products_atom_map_number(zero_based=True)
    bond_map_number = reaction.get_products_bond_map_number(for_changed=True)
    products_g = combine_graphs(product_graphs, atom_map_number, bond_map_number)

    # combine reaction graph from the combined reactant graph and product graph
    num_unchanged = len(reaction.unchanged_bonds)
    num_lost = len(reaction.lost_bonds)
    num_added = len(reaction.added_bonds)

    reaction_g = create_reaction_graph(
        reactants_g, products_g, num_unchanged, num_lost, num_added, num_global_nodes=1
    )

    return reactants_g, products_g, reaction_g, reaction


def assert_atom_subgroup(
    g, sub_g, retained_nodes, retained_bond_edges, need_reorder=False
):
    """
    Args:
        need_reorder: whether to reorder edge feature according to eid. This is needed
        if dgl.node_subgroup() is used, in which the relative order of edges are not
        preserved.
    """

    assert torch.equal(
        g.nodes["global"].data["feat"], sub_g.nodes["global"].data["feat"]
    )
    assert torch.equal(
        g.nodes["atom"].data["feat"][retained_nodes], sub_g.nodes["atom"].data["feat"]
    )
    g_edata = g.edges["bond"].data["feat"][retained_bond_edges]
    sub_g_edata = sub_g.edges["bond"].data["feat"]

    if need_reorder:
        eid = sub_g.edges["bond"].data[dgl.EID].numpy().tolist()
        reorder = [eid.index(i) for i in retained_bond_edges]
        sub_g_edata = sub_g_edata[reorder]

    assert torch.equal(g_edata, sub_g_edata)


def assert_bond_subgroup(g, sub_g, retained_bond_edges):
    assert torch.equal(
        g.nodes["global"].data["feat"], sub_g.nodes["global"].data["feat"]
    )
    assert torch.equal(g.nodes["atom"].data["feat"], sub_g.nodes["atom"].data["feat"])
    assert torch.equal(
        g.edges["bond"].data["feat"][retained_bond_edges],
        sub_g.edges["bond"].data["feat"],
    )
