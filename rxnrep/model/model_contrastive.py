"""
Base model for Constrative representation learning.
"""

from rxnrep.model.model import BaseModel


class BaseContrastiveModel(BaseModel):
    def compute_z(self, mol_graphs, rxn_graphs, metadata):
        nodes = ["atom", "global"]
        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        feats["bond"] = mol_graphs.edges["bond"].data.pop("feat")

        feats, reaction_feats = self(mol_graphs, rxn_graphs, feats, metadata)
        z = self.decode(feats, reaction_feats, metadata)

        return z

    def shared_step(self, batch, mode):

        # ========== compute predictions ==========
        (
            indices,
            (mol_graphs1, mol_graphs2),
            rxn_graphs,
            labels,
            (metadata1, metadata2),
        ) = batch

        # lightning cannot move dgl graphs to gpu, so do it manually
        mol_graphs1 = mol_graphs1.to(self.device)
        mol_graphs2 = mol_graphs2.to(self.device)
        if rxn_graphs is not None:
            rxn_graphs = rxn_graphs.to(self.device)

        z1 = self.compute_z(mol_graphs1, rxn_graphs, metadata1)
        z2 = self.compute_z(mol_graphs2, rxn_graphs, metadata2)
        preds = {"z1": z1, "z2": z2}

        # ========== compute losses ==========
        all_loss = self.compute_loss(preds, labels)

        # ========== log the loss ==========
        total_loss = sum(all_loss.values())

        self.log_dict(
            {f"{mode}/loss/{task}": loss for task, loss in all_loss.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            f"{mode}/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        return total_loss, preds, labels, indices
