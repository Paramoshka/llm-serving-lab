from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.model import CharacterGRU


class CharacterDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, token_ids: torch.Tensor, sequence_length: int) -> None:
        self.token_ids = token_ids
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.sequence_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = start + self.sequence_length
        return self.token_ids[start:end], self.token_ids[start + 1 : end + 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny model on a text file")
    parser.add_argument("--data", type=Path, required=True, help="Path to a UTF-8 text file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.pt"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested, but CUDA is unavailable. "
            "Install a CUDA-enabled PyTorch build or run with --device cpu."
        )
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if (
        args.epochs < 1
        or args.sequence_length < 1
        or args.learning_rate <= 0
        or args.max_chars < 1
        or args.batch_size < 1
    ):
        raise ValueError(
            "epochs, sequence-length, max-chars, and batch-size must be greater than zero; "
            "learning-rate must be positive"
        )

    text = args.data.read_text(encoding="utf-8")[: args.max_chars]
    vocabulary = sorted(set(text))
    if len(text) <= args.sequence_length or len(vocabulary) < 2:
        raise ValueError("Text must be longer than sequence-length and contain at least two distinct characters")

    char_to_id = {char: index for index, char in enumerate(vocabulary)}
    token_ids = torch.tensor([char_to_id[char] for char in text], dtype=torch.long)
    device = resolve_device(args.device)
    dataset = CharacterDataset(token_ids, args.sequence_length)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    model = CharacterGRU(len(vocabulary)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_tokens = 0
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = loss_function(logits.reshape(-1, len(vocabulary)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
        print(f"epoch {epoch + 1}/{args.epochs}: loss={total_loss / total_tokens:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            },
            "vocabulary": vocabulary,
            "embedding_size": 32,
            "hidden_size": 64,
        },
        args.output,
    )
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
