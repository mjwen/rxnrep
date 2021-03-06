import numpy as np
import torch

from rxnrep.data.transformer import GraphFeatureTransformer, StandardScaler
from rxnrep.tests.data.utils import create_graph_C, create_graph_CO2


def test_stadard_scaler():
    data = torch.arange(15).float().reshape(5, 3)

    mean = torch.mean(data, dim=0)
    std = torch.from_numpy(np.std(data.numpy(), axis=0))
    ref = (data - mean) / std

    transformer = StandardScaler()
    trans_data = transformer.transform(data)
    state_dict = transformer.state_dict()

    assert torch.equal(mean, state_dict["mean"])
    assert torch.equal(std, state_dict["std"])
    assert torch.equal(ref, trans_data)


def test_graph_feature_transformer():
    def get_feats(graphs, nv):
        atom_feats = [g.nodes["atom"].data["feat"] for g in graphs]
        bond_feats = [g.edges["a2a"].data["feat"] for g in graphs]
        if nv > 0:
            virtual_feats = [g.nodes["virtual"].data["feat"] for g in graphs]
        else:
            virtual_feats = None

        feats = {"atom": atom_feats, "bond": bond_feats, "virtual": virtual_feats}

        return feats

    nv = 1

    graphs = [
        create_graph_C(num_virtual_nodes=nv),
        create_graph_CO2(num_virtual_nodes=nv),
    ]

    feats = get_feats(graphs, nv)
    mean = {
        k: torch.mean(torch.cat(v), dim=0) for k, v in feats.items() if v is not None
    }

    # we use numpy because torch std and numpy std is a bit different and numpy std is
    # used in transformer
    std = {
        k: torch.from_numpy(np.std(torch.cat(v).numpy(), axis=0))
        for k, v in feats.items()
        if v is not None
    }
    ref_value = {}
    for k, ft in feats.items():
        if ft is not None:
            n = (torch.cat(ft) - mean[k]) / std[k]
            ref_value[k] = torch.split(n, [len(f) for f in ft])

    transformer = GraphFeatureTransformer()
    graphs = transformer.transform(graphs)
    feats = get_feats(graphs, nv)

    state_dict = transformer.state_dict()
    assert torch.equal(state_dict["mean"]["node"]["atom"], mean["atom"])
    assert torch.equal(state_dict["std"]["node"]["atom"], std["atom"])
    assert torch.equal(state_dict["mean"]["edge"]["a2a"], mean["bond"])
    assert torch.equal(state_dict["std"]["edge"]["a2a"], std["bond"])
    if nv > 0:
        assert torch.equal(state_dict["mean"]["node"]["virtual"], mean["virtual"])
        # Transformer set std to 1 if it is actually 0
        assert torch.equal(
            state_dict["std"]["node"]["virtual"], torch.tensor([1.0, 1.0])
        )

    for k, ft in feats.items():
        if k != "virtual":
            ref = ref_value[k]
            for f, r in zip(ft, ref):
                assert torch.equal(f, r)