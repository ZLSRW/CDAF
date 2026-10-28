"""Type-specific feature encoders and attention over original disease edges."""

import math
import torch
from torch import nn


def feature_encoder(input_dim, hidden_dim, dropout):
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
    )


class DiseaseContext(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.empty = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, entities, diseases, edges):
        n, h = entities.shape
        if not len(edges):
            return self.empty.expand(n, -1), entities.new_zeros(n)
        source, disease = edges.unbind(dim=1)
        scores = (self.query(entities)[source] * self.key(diseases)[disease]).sum(-1) / math.sqrt(h)
        maximum = scores.new_full((n,), -torch.inf)
        maximum.scatter_reduce_(0, source, scores.detach(), reduce="amax", include_self=True)
        weights = (scores - maximum[source]).exp()
        denominator = scores.new_zeros(n).index_add(0, source, weights)
        weights = weights / denominator[source].clamp_min(1e-12)
        values = self.value(diseases)[disease] * weights[:, None]
        context = entities.new_zeros(n, h).index_add(0, source, values)
        available = denominator > 0
        context = torch.where(available[:, None], context, self.empty[None, :])
        return context, available.to(entities.dtype)


class InputEncoders(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.drug = feature_encoder(300, hidden_dim, dropout)
        self.protein = feature_encoder(320, hidden_dim, dropout)
        self.disease = feature_encoder(64, hidden_dim, dropout)
        self.drug_context = DiseaseContext(hidden_dim)
        self.protein_context = DiseaseContext(hidden_dim)

    def forward(self, drug, protein, disease, ds, ps):
        hd, hp, hs = self.drug(drug), self.protein(protein), self.disease(disease)
        return self.condition_features(dict(drug=hd, protein=hp, disease=hs), ds, ps)

    def condition_features(self, encoded, ds, ps):
        """Recompute trainable conditions from possibly cached, frozen base features."""
        hd, hp, hs = (encoded[k] for k in ("drug", "protein", "disease"))
        cd, md = self.drug_context(hd, hs, ds)
        cp, mp = self.protein_context(hp, hs, ps)
        return dict(
            drug=hd,
            protein=hp,
            disease=hs,
            drug_context=cd,
            protein_context=cp,
            drug_available=md,
            protein_available=mp,
        )
