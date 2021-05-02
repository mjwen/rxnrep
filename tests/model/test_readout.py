import dgl
import torch
from dgl.nn import Set2Set as Set2SetDGL

from rxnrep.model.readout import Set2Set
from rxnrep.utils.io import seed_all


def create_graph_CO2():
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges(u=[0, 1, 1, 2], v=[1, 0, 2, 1])

    g.ndata["feat"] = torch.arange(9).float().reshape(3, 3)

    return g


def create_graph_C():
    g = dgl.DGLGraph()
    g.add_nodes(1)
    g.ndata["feat"] = torch.arange(3).float().reshape(1, 3)

    return g


def test_set2set_without_graph():
    """
    Check our implementation against dgl implementation.
    """
    graphs = [create_graph_C(), create_graph_CO2()]

    bg = dgl.batch([graphs[1], graphs[1], graphs[0]])
    feats = bg.ndata["feat"]

    seed_all()
    s2s_wo = Set2Set(input_dim=3, n_iters=2, n_layers=2)
    results_wo = s2s_wo(feats, bg.batch_num_nodes())

    seed_all()
    s2s_w = Set2SetDGL(input_dim=3, n_iters=2, n_layers=2)
    results_w = s2s_w(bg, feats)

    assert torch.equal(results_wo, results_w)
