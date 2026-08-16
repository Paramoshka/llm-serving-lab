from pathlib import Path

import pytest

from src.download_data import format_dialogue, write_dialogues


def test_format_dialogue_assigns_alternating_speakers() -> None:
    result = format_dialogue(["Hello", "Hi there", "How are you?"])

    assert result == "User: Hello\nAssistant: Hi there\nUser: How are you?"


def test_write_dialogues_rejects_invalid_rows_without_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "dialogues.txt"

    with pytest.raises(ValueError, match="dialog"):
        write_dialogues([{"utterances": "not a list"}], output)

    assert not output.exists()
