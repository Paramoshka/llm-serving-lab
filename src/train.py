from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.model import CharacterGRU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny model on a text file")
    parser.add_argument("--data", type=Path, required=True, help="Path to a UTF-8 text file")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.sequence_length < 1 or args.learning_rate <= 0:
        raise ValueError("epochs and sequence-length must be greater than zero; learning-rate must be positive")

    text = args.data.read_text(encoding="utf-8")
    vocabulary = sorted(set(text))
    if len(text) <= args.sequence_length or len(vocabulary) < 2:
        raise ValueError("Text must be longer than sequence-length and contain at least two distinct characters")

    char_to_id = {char: index for index, char in enumerate(vocabulary)}
    token_ids = torch.tensor([char_to_id[char] for char in text], dtype=torch.long)
    inputs = token_ids[:-1].unfold(0, args.sequence_length, 1)
    targets = token_ids[1:].unfold(0, args.sequence_length, 1)

    model = CharacterGRU(len(vocabulary))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = loss_function(logits.reshape(-1, len(vocabulary)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch + 1}/{args.epochs}: loss={loss.item():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "embedding_size": 32,
            "hidden_size": 64,
        },
        args.output,
    )
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
