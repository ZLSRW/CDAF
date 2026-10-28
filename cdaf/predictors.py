"""Single prediction head used when ablating the entire evidence fusion."""

import torch
from torch import nn


class SinglePairPredictor(nn.Module):
    """Predict from drug/protein features without graph, path, or channel routing."""

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

    def forward(self, drug, protein):
        features = torch.cat((drug, protein, drug * protein, (drug - protein).abs()), dim=-1)
        return self.network(features).squeeze(-1)
