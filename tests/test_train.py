import sys
from pathlib import Path

import pytest
import torch

from src import train


def test_resolve_cuda_requires_available_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA training was requested"):
        train.resolve_device("cuda")


def test_training_saves_cpu_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "train.txt"
    output = tmp_path / "model.pt"
    data.write_text("hello hello hello hello hello hello", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--data",
            str(data),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--sequence-length",
            "4",
            "--batch-size",
            "2",
        ],
    )

    train.main()

    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert output.exists()
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["state_dict"].values())
