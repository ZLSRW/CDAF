"""Bilinear interaction, directed relational graph and explicit disease path channels."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class BilinearChannel(nn.Module):
    def __init__(self, hidden_dim, heads, rank):
        super().__init__()
        self.heads, self.rank = heads, rank
        self.drug = nn.Linear(hidden_dim, heads * rank, bias=False)
        self.protein = nn.Linear(hidden_dim, heads * rank, bias=False)
        self.head_score = nn.Parameter(torch.empty(heads, rank))
        nn.init.normal_(self.head_score, std=1 / math.sqrt(rank))
        self.classifier = nn.Linear(rank, 1)

    def forward(self, drug, protein):
        left = F.silu(self.drug(drug)).reshape(-1, self.heads, self.rank)
        right = F.silu(self.protein(protein)).reshape(-1, self.heads, self.rank)
        interaction = left * right
        weights = (interaction * self.head_score).sum(-1).softmax(-1)
        representation = (weights[:, :, None] * interaction).sum(1)
        return self.classifier(representation).squeeze(-1), representation, weights


class RelationalLayer(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.self_drug = nn.Linear(hidden_dim, hidden_dim)
        self.self_protein = nn.Linear(hidden_dim, hidden_dim)
        self.drug_to_protein = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.protein_to_drug = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm_drug, self.norm_protein = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug, protein, edges, drug_degree, protein_degree):
        d, p = edges.unbind(dim=1)
        # index_add accumulates every original edge record, including repetitions.
        dm = torch.zeros_like(drug).index_add(0, d, self.protein_to_drug(protein)[p])
        pm = torch.zeros_like(protein).index_add(0, p, self.drug_to_protein(drug)[d])
        dm = dm / drug_degree[:, None].clamp_min(1)
        pm = pm / protein_degree[:, None].clamp_min(1)
        return (
            self.norm_drug(drug + self.dropout(F.gelu(self.self_drug(drug) + dm))),
            self.norm_protein(protein + self.dropout(F.gelu(self.self_protein(protein) + pm))),
        )


class GraphChannel(nn.Module):
    def __init__(self, hidden_dim, layers, rank, dropout):
        super().__init__()
        self.layers = nn.ModuleList([RelationalLayer(hidden_dim, dropout) for _ in range(layers)])
        self.hop_weights = nn.Parameter(torch.zeros(2, layers + 1))
        self.drug = nn.Linear(hidden_dim, rank, bias=False)
        self.protein = nn.Linear(hidden_dim, rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))
        self.rank = rank

    def encode(self, drug, protein, edges):
        ddeg = torch.bincount(edges[:, 0], minlength=len(drug)).to(drug.dtype)
        pdeg = torch.bincount(edges[:, 1], minlength=len(protein)).to(protein.dtype)
        dh, ph = [drug], [protein]
        for layer in self.layers:
            drug, protein = layer(drug, protein, edges, ddeg, pdeg)
            dh.append(drug)
            ph.append(protein)
        weights = self.hop_weights.softmax(-1)
        drug = sum(weight * value for weight, value in zip(weights[0], dh))
        protein = sum(weight * value for weight, value in zip(weights[1], ph))
        return drug, protein, ddeg, pdeg

    def decode(self, drug, protein):
        representation = self.drug(drug) * self.protein(protein)
        return representation.sum(-1) / math.sqrt(self.rank) + self.bias, representation


class PathChannel(nn.Module):
    def __init__(self, hidden_dim, heads, dropout, degree_penalty):
        super().__init__()
        self.cls = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.types = nn.Parameter(torch.empty(1, 4, hidden_dim))
        self.positions = nn.Parameter(torch.empty(1, 4, hidden_dim))
        for parameter in [self.cls, self.types, self.positions]:
            nn.init.normal_(parameter, std=0.02)
        self.transformer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 2, dropout, batch_first=True, norm_first=True
        )
        self.query_drug = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_protein = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.degree_penalty, self.hidden_dim = degree_penalty, hidden_dim

    def forward(self, drug, protein, diseases, path_ids, path_valid, disease_degree):
        batch, max_paths = path_ids.shape
        rows, slots = path_valid.nonzero(as_tuple=True)
        representations = drug.new_zeros(batch, max_paths, self.hidden_dim)
        if len(rows):
            mids = diseases[path_ids[rows, slots]]
            tokens = torch.stack((drug[rows], mids, protein[rows]), dim=1)
            tokens = torch.cat((self.cls.expand(len(rows), -1, -1), tokens), dim=1)
            encoded = self.transformer(tokens + self.types + self.positions)[:, 0]
            representations[rows, slots] = encoded
        query = self.query_drug(drug) * self.query_protein(protein)
        scores = (self.key(representations) * query[:, None, :]).sum(-1) / math.sqrt(
            self.hidden_dim
        )
        scores = scores - self.degree_penalty * torch.log1p(disease_degree[path_ids])
        available = path_valid.any(-1)
        scores = scores.masked_fill(~path_valid, -torch.inf)
        # Avoid softmax(all -inf) for candidates without a shared disease.
        scores = torch.where(available[:, None], scores, torch.zeros_like(scores))
        weights = scores.softmax(-1) * path_valid.to(scores.dtype)
        representation = (representations * weights[:, :, None]).sum(1)
        return self.classifier(representation).squeeze(-1), representation, weights, available
