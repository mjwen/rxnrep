import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from rxnrep.layer.encoder import ReactionEncoder, adjust_encoder_config
from rxnrep.layer.utils import MLP
from rxnrep.model.model_contrastive import BaseContrastiveModel
from rxnrep.utils.config import determine_layer_size_by_pool_method, merge_configs


def adjust_decoder_config(config: DictConfig) -> DictConfig:
    size = determine_layer_size_by_pool_method(config.model.encoder)
    minimum = config.model.encoder.conv_layer_size

    num_layers = config.model.decoder.model_class.simclr_decoder_num_layers
    layer_sizes = [max(size // 2 ** i, minimum) for i in range(num_layers)]

    new_config = OmegaConf.create(
        {
            "model": {
                "decoder": {"model_class": {"simclr_decoder_layer_sizes": layer_sizes}}
            }
        }
    )

    return new_config


def adjust_config(config: DictConfig) -> DictConfig:
    """
    Adjust model config, both encoder and decoder.
    """
    encoder_config = adjust_encoder_config(config)

    # create a temporary one to merge original and encoder
    # this is needed since info from both config is needed to adjust decoder config
    merged = merge_configs(config, encoder_config)
    decoder_config = adjust_decoder_config(merged)
    model_config = merge_configs(encoder_config, decoder_config)

    return model_config


class LightningModel(BaseContrastiveModel):
    def init_backbone(self, params):

        model = ReactionEncoder(
            in_feats=params.dataset_info["feature_size"],
            embedding_size=params.embedding_size,
            molecule_conv_layer_sizes=params.molecule_conv_layer_sizes,
            molecule_num_fc_layers=params.molecule_num_fc_layers,
            molecule_batch_norm=params.molecule_batch_norm,
            molecule_activation=params.activation,
            molecule_residual=params.molecule_residual,
            molecule_dropout=params.molecule_dropout,
            conv=params.conv,
            combine_reactants_products=params.combine_reactants_products,
        )

        return model

    def init_decoder(self, params):
        projection_decoder = MLP(
            in_size=self.backbone.reaction_feats_size,
            hidden_sizes=params.simclr_decoder_layer_sizes[:-1],
            activation=params.activation,
            out_size=params.simclr_decoder_layer_sizes[-1],
        )

        return {"projection_head_decoder": projection_decoder}

    def decode(self, feats, reaction_feats, metadata):
        decoder = self.decoder["projection_head_decoder"]
        z = decoder(reaction_feats)

        return z

    def compute_loss(self, preds, labels):
        z1 = preds["z1"]
        z2 = preds["z2"]

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        loss = nt_xent_loss(z1, z2, self.hparams.simclr_temperature)

        return {"contrastive": loss}


def nt_xent_loss(out_1, out_2, temperature, eps=1e-6):
    """
    assume out_1 and out_2 are normalized
    out_1: [batch_size, dim]
    out_2: [batch_size, dim]

    From: https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py
    """
    # gather representations in case of distributed training
    # out_1_dist: [batch_size * world_size, dim]
    # out_2_dist: [batch_size * world_size, dim]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        out_1_dist = SyncFunction.apply(out_1)
        out_2_dist = SyncFunction.apply(out_2)
    else:
        out_1_dist = out_1
        out_2_dist = out_2

    # out: [2 * batch_size, dim]
    # out_dist: [2 * batch_size * world_size, dim]
    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
    # neg: [2 * batch_size]
    cov = torch.mm(out, out_dist.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^1 to remove similarity measure for x1.x1
    row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()

    return loss


class SyncFunction(torch.autograd.Function):
    """
    From: https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py
    """

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(
            grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False
        )

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]
