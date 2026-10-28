"""Disease-conditioned latent refinement and availability-aware channel fusion."""

import torch
from torch import nn
from torch.nn import functional as F

from .encoders import InputEncoders
from .diffusion import LatentDiffusion
from .channels import BilinearChannel, GraphChannel, PathChannel
from .predictors import SinglePairPredictor


class CDAF(nn.Module):
    """The fixed entity-diffusion architecture evaluated in the manuscript."""

    def __init__(self, args):
        super().__init__()
        h, rank = args.hidden_dim, args.bilinear_rank
        self.encoders = InputEncoders(h, args.dropout)
        self.enhancement = "entity"
        self.diffusion = LatentDiffusion(args)
        self.single_predictor = getattr(args, "single_predictor", False)
        self.disable_diffusion = args.disable_diffusion
        self.train_condition_encoders = (
            getattr(args, "train_condition_encoders", False) and not self.disable_diffusion
        )
        self.encoders_frozen = False
        if args.disable_diffusion:
            # Keep the initialization order fixed across ablation conditions.
            self.diffusion.requires_grad_(False)
        if self.single_predictor:
            self.predictor = SinglePairPredictor(h, args.dropout)
            return
        self.bilinear = BilinearChannel(h, args.bilinear_heads, rank)
        self.graph = GraphChannel(h, args.graph_layers, rank, args.dropout)
        self.path = PathChannel(h, args.path_heads, args.dropout, args.path_degree_penalty)
        self.router_projections = nn.ModuleList(
            [nn.Linear(rank, 64), nn.Linear(rank, 64), nn.Linear(h, 64)]
        )
        # Three channel representations, pair product, two degrees, path count,
        # and two disease-context availability indicators.
        self.router = nn.Sequential(nn.Linear(192 + h + 5, h), nn.GELU(), nn.Linear(h, 3))
        self.disable_graph, self.disable_path = args.disable_graph, args.disable_path
        self.uniform_gate = args.uniform_gate

    def train(self, mode=True):
        super().train(mode)
        if self.encoders_frozen:
            self.encoders.eval()
        return self

    def freeze_encoders(self):
        self.encoders_frozen = True
        # Stale warmup gradients must not enter subsequent gradient clipping.
        self.encoders.zero_grad(set_to_none=True)
        self.encoders.requires_grad_(False)
        if self.train_condition_encoders:
            self.encoders.drug_context.requires_grad_(True)
            self.encoders.protein_context.requires_grad_(True)
        self.encoders.eval()

    def condition_parameters(self):
        return list(self.encoders.drug_context.parameters()) + list(
            self.encoders.protein_context.parameters()
        )

    def refresh_conditions(self, encoded, tensors):
        if encoded is not None and self.encoders_frozen and self.train_condition_encoders:
            return self.encoders.condition_features(encoded, tensors["ds"], tensors["ps"])
        return encoded

    def encode(self, tensors):
        return self.encoders(*(tensors[k] for k in ("drug", "protein", "disease", "ds", "ps")))

    def enhance(self, encoded, use_diffusion, seed=None):
        if not use_diffusion or self.disable_diffusion:
            return encoded["drug"], encoded["protein"]
        generator = None
        if seed is not None:
            generator = torch.Generator(device=encoded["drug"].device).manual_seed(seed)
        outputs = []
        for kind, name in enumerate(("drug", "protein")):
            x = encoded[name]
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            outputs.append(
                self.diffusion.enhance(
                    x, encoded[name + "_context"], encoded[name + "_available"], kind, noise
                )
            )
        return tuple(outputs)

    def prepare_graph(self, enhanced, edges):
        if self.single_predictor:
            return None
        if self.disable_graph:
            drug, protein = enhanced
            return drug, protein, drug.new_zeros(len(drug)), protein.new_zeros(len(protein))
        return self.graph.encode(*enhanced, edges)

    def score(
        self,
        pairs,
        encoded,
        enhanced,
        graph_state,
        path_data,
        disease_degree,
        use_diffusion=True,
        noise_seed=None,
    ):
        d, p = pairs.unbind(dim=1)
        drug, protein = enhanced[0][d], enhanced[1][p]
        if self.single_predictor:
            logit = self.predictor(drug, protein)
            return dict(probability=logit.sigmoid(), single_logit=logit)
        bscore, brep, head_weights = self.bilinear(drug, protein)
        gd, gp, ddeg, pdeg = graph_state
        gscore, grep = self.graph.decode(gd[d], gp[p])
        graph_available = (ddeg[d] + pdeg[p]) > 0
        path_ids, path_valid, path_count = path_data
        if self.disable_path:
            path_valid = torch.zeros_like(path_valid)
            path_count = torch.zeros_like(path_count)
        pscore, prep, path_weights, path_available = self.path(
            drug, protein, encoded["disease"], path_ids, path_valid, disease_degree
        )
        available = torch.stack(
            (torch.ones_like(graph_available), graph_available, path_available), dim=1
        )
        representations = [
            F.gelu(projection(rep))
            for projection, rep in zip(self.router_projections, (brep, grep, prep))
        ]
        representations = [rep * available[:, i, None] for i, rep in enumerate(representations)]
        evidence = torch.stack(
            (
                torch.log1p(ddeg[d]),
                torch.log1p(pdeg[p]),
                torch.log1p(path_count.to(drug.dtype)),
                encoded["drug_available"][d],
                encoded["protein_available"][p],
            ),
            dim=1,
        )
        gate_logits = self.router(torch.cat((*representations, drug * protein, evidence), dim=1))
        if self.uniform_gate:
            gate_logits = torch.zeros_like(gate_logits)
        weights = gate_logits.masked_fill(~available, -torch.inf).softmax(dim=1)
        logits = torch.stack((bscore, gscore, pscore), dim=1)
        probabilities = (weights * logits.sigmoid()).sum(dim=1)
        return dict(
            probability=probabilities,
            channel_logits=logits,
            gate=weights,
            available=available,
            head_weights=head_weights,
            path_weights=path_weights,
            path_ids=path_ids,
        )

    def forward(
        self,
        tensors,
        pairs,
        context_edges,
        path_data,
        disease_degree,
        use_diffusion=True,
        encoded=None,
        noise_seed=None,
    ):
        if encoded is None:
            encoded = self.encode(tensors)
        enhanced = self.enhance(encoded, use_diffusion, noise_seed)
        graph_state = self.prepare_graph(enhanced, context_edges)
        return self.score(
            pairs,
            encoded,
            enhanced,
            graph_state,
            path_data,
            disease_degree,
            use_diffusion,
            noise_seed,
        )


def classification_loss(output, labels, auxiliary_weight):
    """Prediction BCE plus mean BCE over the available evidence channels."""
    if "single_logit" in output:
        return F.binary_cross_entropy_with_logits(output["single_logit"], labels)
    main = F.binary_cross_entropy(output["probability"].clamp(1e-7, 1 - 1e-7), labels)
    auxiliary = []
    for index in range(3):
        valid = output["available"][:, index]
        if valid.any():
            auxiliary.append(
                F.binary_cross_entropy_with_logits(
                    output["channel_logits"][valid, index], labels[valid]
                )
            )
    return main + auxiliary_weight * torch.stack(auxiliary).mean()
