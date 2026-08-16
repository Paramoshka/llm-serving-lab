# llm-serving-lab

A CPU-first sandbox for training and serving a tiny PyTorch model.

The model is a character-level GRU that learns to predict the next character. It is not a full LLM, but it is the shortest practical path through the complete workflow: data → training → API → Docker → Kubernetes.

## Lab goals

This lab will evolve from the CPU baseline into a production-style LLM serving exercise:

- Serve an open model with vLLM.
- Send requests through an OpenAI-compatible API.
- Deploy the service to Kubernetes.
- Observe model and GPU metrics.
- Load the service with concurrent requests and explain latency growth.
- Compare two vLLM configurations.
- Recover from a Pod failure and update the model.
- Estimate request costs.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
printf 'a tiny model learns to continue text.\n' > data/input.txt
python -m src.train --data data/input.txt --epochs 20
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

Check the service:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"tiny", "max_new_tokens":30}'
```

## Docker

Train the model first so that `artifacts/model.pt` is included in the image. The image can also be built before training, but then `/health` returns `model_not_loaded` and `/generate` returns `503`. Build and run the service:

```bash
docker build -t mini-llm:local .
docker run --rm -p 8000:8000 mini-llm:local
```

For Kubernetes, push the image to a registry available to the cluster and replace `image` in `kubernetes/model/deployment.yaml`.

## Project layout

```text
benchmarks/       latency, throughput, and quality measurements
dashboards/       future dashboard definitions
docs/             experiment notes
kubernetes/model/ CPU service deployment
load-generator/   future load generator
src/              model, training, and HTTP API
data/             local training texts
artifacts/        local checkpoints (not committed)
```
