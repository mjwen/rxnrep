import dgl
import torch

from rxnrep.core.reaction import smiles_to_reaction
from rxnrep.data.to_graph import combine_graphs, create_reaction_graph
from rxnrep.data.transforms import DropAtom
from rxnrep.tests.utils import create_graph
from rxnrep.utils import seed_all


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
    retained_bond_edge = [2, 3, 8, 9]

    g1 = reactants_g
    g2 = sub_reactants_g
    assert torch.equal(g1.nodes["global"].data["feat"], g2.nodes["global"].data["feat"])
    assert torch.equal(
        g1.nodes["atom"].data["feat"][select], g2.nodes["atom"].data["feat"]
    )
    eid = g2.edges["bond"].data[dgl.EID].numpy().tolist()
    edata = g1.edges["bond"].data["feat"][retained_bond_edge]
    reorder = [eid.index(i) for i in retained_bond_edge]
    assert torch.equal(edata, g2.edges["bond"].data["feat"][reorder])

    g1 = products_g
    g2 = sub_products_g
    assert torch.equal(g1.nodes["global"].data["feat"], g2.nodes["global"].data["feat"])
    assert torch.equal(
        g1.nodes["atom"].data["feat"][select], g2.nodes["atom"].data["feat"]
    )
    eid = g2.edges["bond"].data[dgl.EID].numpy().tolist()
    edata = g1.edges["bond"].data["feat"][retained_bond_edge]
    reorder = [eid.index(i) for i in retained_bond_edge]
    assert torch.equal(edata, g2.edges["bond"].data["feat"][reorder])
