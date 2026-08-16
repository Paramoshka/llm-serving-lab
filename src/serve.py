from __future__ import annotations

from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import CharacterGRU

MODEL_PATH = Path("artifacts/model.pt")
app = FastAPI(title="Mini LLM", version="0.1.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    max_new_tokens: int = Field(default=50, ge=1, le=500)
    temperature: float = Field(default=1.0, gt=0, le=2)


def load_model(path: Path) -> tuple[CharacterGRU, list[str], dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train the model first.")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    vocabulary = checkpoint["vocabulary"]
    model = CharacterGRU(
        len(vocabulary), checkpoint["embedding_size"], checkpoint["hidden_size"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocabulary, {char: index for index, char in enumerate(vocabulary)}


try:
    model, vocabulary, char_to_id = load_model(MODEL_PATH)
except FileNotFoundError:
    model = None
    vocabulary = []
    char_to_id = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if model is not None else "model_not_loaded"}


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, str]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    unknown = set(request.prompt) - set(char_to_id)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown characters: {''.join(sorted(unknown))}")

    token_ids = [char_to_id[char] for char in request.prompt]
    with torch.inference_mode():
        logits, hidden = model(torch.tensor([token_ids]))
        generated = list(request.prompt)
        for index in range(request.max_new_tokens):
            probabilities = torch.softmax(logits[0, -1] / request.temperature, dim=0)
            current_token = torch.multinomial(probabilities, num_samples=1).reshape(1, 1)
            generated.append(vocabulary[current_token.item()])
            if index < request.max_new_tokens - 1:
                logits, hidden = model(current_token, hidden)

    return {"text": "".join(generated)}
