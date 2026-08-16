from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path


DEFAULT_DATASET = "roskoN/dailydialog"


def format_dialogue(dialogue: Iterable[str]) -> str:
    lines = []
    for index, utterance in enumerate(dialogue):
        speaker = "User" if index % 2 == 0 else "Assistant"
        text = utterance.strip()
        if text:
            lines.append(f"{speaker}: {text}")

    if len(lines) < 2:
        raise ValueError("Each dialogue must contain at least two non-empty utterances")
    return "\n".join(lines)


def write_dialogues(rows: Iterable[Mapping[str, object]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(f"{output.suffix}.tmp")
    count = 0

    try:
        with temporary_output.open("w", encoding="utf-8") as file:
            for row in rows:
                dialogue = row.get("utterances")
                if not isinstance(dialogue, list) or not all(
                    isinstance(utterance, str) for utterance in dialogue
                ):
                    raise ValueError("Dataset rows must contain an 'utterances' list of strings")
                file.write(format_dialogue(dialogue))
                file.write("\n\n")
                count += 1
        temporary_output.replace(output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize the DailyDialog dataset")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("data/dailydialog.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Install requirements-train.txt before downloading the dataset") from error

    dataset = load_dataset(args.dataset, split=args.split)
    count = write_dialogues(dataset, args.output)
    print(f"Saved {count} dialogues to {args.output}")


if __name__ == "__main__":
    main()
