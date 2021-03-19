import dgl
import torch

from rxnrep.core.reaction import smiles_to_reaction
from rxnrep.data.to_graph import combine_graphs, create_reaction_graph
from rxnrep.data.transforms_batch import DropAtomBatch
from rxnrep.utils import seed_all

from ..utils import create_graph


def test_drop_atom():
    seed_all(25)

    reactants_g, products_g, reaction_g, reaction = create_reaction_and_graphs()

    batch_reactants_g = dgl.batch([reactants_g, reactants_g])
    batch_products_g = dgl.batch([products_g, products_g])
    reactions = [reaction, reaction]

    transform = DropAtomBatch(ratio=0.5)
    sub_reactants_g, sub_products_g, sub_reaction_g, _ = transform(
        batch_reactants_g, batch_products_g, None, reactions
    )

    # A total number of 6 atoms in each graph. Atom 3 is dropped in graph 1 and atom 1
    # is dropped in graph 2.
    select_1 = [True, True, True, False, True, True]
    select_2 = [True, False, True, True, True, True]
    retained_nodes = select_1 + select_2

    # A total number of 10 edges in each graph.  Bonds 1 and 4 are kept after dropping
    # atom 3 in graph 1; bonds 2, 3, and 4 are kept after dropping atom 1 in graph 2.
    retained_bond_edges = [2, 3, 8, 9] + [14, 15, 16, 17, 18, 19]

    assert_atom_subgroup(
        batch_reactants_g, sub_reactants_g, retained_nodes, retained_bond_edges
    )
    assert_atom_subgroup(
        batch_products_g, sub_products_g, retained_nodes, retained_bond_edges
    )

    # assert batch info
    a2a = ("atom", "bond", "atom")
    a2g = ("atom", "a2g", "global")
    g2a = ("global", "g2a", "atom")
    assert torch.equal(sub_reactants_g.batch_num_edges(a2a), torch.tensor([4, 6]))
    assert torch.equal(sub_reactants_g.batch_num_edges(a2g), torch.tensor([5, 5]))
    assert torch.equal(sub_reactants_g.batch_num_edges(g2a), torch.tensor([5, 5]))
    assert torch.equal(sub_reactants_g.batch_num_nodes("atom"), torch.tensor([5, 5]))
    assert torch.equal(sub_reactants_g.batch_num_nodes("global"), torch.tensor([2, 2]))


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
    eid = sub_g.edges["bond"].data[dgl.EID].numpy().tolist()
    edata = g.edges["bond"].data["feat"][retained_bond_edges]
    reorder = [eid.index(i) for i in retained_bond_edges]
    assert torch.equal(edata, sub_g.edges["bond"].data["feat"][reorder])
