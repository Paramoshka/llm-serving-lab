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

## GPU training

```bash
python -m venv .venv
source .venv/bin/activate
# Install a CUDA-enabled PyTorch wheel for your NVIDIA driver from pytorch.org.
python -m pip install -r requirements.txt -r requirements-train.txt
make data
make learn
```

`make learn` uses CUDA by default. For an explicit CPU fallback, run `make learn DEVICE=cpu`.
The default profile uses the first 200,000 characters, batch size 128, and 10 epochs. Override it as needed, for example: `make learn EPOCHS=20 MAX_CHARS=1000000`.

DailyDialog is downloaded only to `data/dailydialog.txt`, which is ignored by Git. It is licensed under [CC BY-NC-SA 4.0](https://huggingface.co/datasets/roskoN/dailydialog), so use it only for this non-commercial lab and retain attribution when sharing derivatives.

## Tests

```bash
python -m pip install -r requirements-dev.txt
make test
```

## Build and run

`make build` runs training first and then builds a CPU-only image that includes `artifacts/model.pt`:

```bash
make build
docker run --rm -p 8000:8000 mini-llm:local
```

Check the service:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"User: hello\nAssistant:", "max_new_tokens":30}'
```

For Kubernetes, push `mini-llm:local` to a registry available to the cluster and replace `image` in `kubernetes/model/deployment.yaml`.

## Project layout

```text
benchmarks/       latency, throughput, and quality measurements
dashboards/       future dashboard definitions
docs/             experiment notes
kubernetes/model/ CPU service deployment
load-generator/   future load generator
src/              model, training, and HTTP API
data/             locally downloaded training data
artifacts/        local checkpoints (not committed)
```
