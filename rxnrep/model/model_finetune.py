"""
Base model for finetuning.
"""

import torch

from rxnrep.model.model import BaseModel


class BaseFinetuneModel(BaseModel):
    def configure_optimizers(self):
        """
        Different learning rate for prediction head and backbone encoder (if it is
        requested to be optimized).

        """

        # params for prediction head decoder
        assert (
            len(self.decoder) == 1
        ), f"Expect 1 decoder for finetune model, got {len(self.decoder)}"

        prediction_head = list(self.decoder.values())[0]

        params_group = [
            {
                "params": filter(
                    lambda p: p.requires_grad, prediction_head.parameters()
                ),
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay,
            }
        ]

        # params in encoder
        if self.hparams.finetune_tune_encoder:
            params_group.append(
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.backbone.parameters()
                    ),
                    "lr": self.hparams.finetune_lr_encoder,
                    "weight_decay": self.hparams.weight_decay,
                }
            )

        optimizer = torch.optim.Adam(params_group)

        # learning rate scheduler
        scheduler = self._config_lr_scheduler(optimizer)

        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val/score",
            }
