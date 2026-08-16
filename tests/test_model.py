import torch

from src.model import CharacterGRU


def test_model_returns_logits_for_every_token() -> None:
    model = CharacterGRU(vocab_size=5, embedding_size=4, hidden_size=8)

    logits, hidden = model(torch.tensor([[0, 1, 2]]))

    assert logits.shape == (1, 3, 5)
    assert hidden.shape == (1, 1, 8)
