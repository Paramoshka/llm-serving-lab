from __future__ import annotations

import torch
from torch import nn


class CharacterGRU(nn.Module):
    """Minimal language model that predicts the next character."""

    def __init__(self, vocab_size: int, embedding_size: int = 32, hidden_size: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, tokens: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(tokens)
        encoded, hidden = self.gru(embedded, hidden)
        return self.output(encoded), hidden
