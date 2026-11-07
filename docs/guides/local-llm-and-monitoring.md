# Local LLMs + GPU monitoring

BankAI talks to local engines through **LiteLLM** (OpenAI-compatible `/v1`). The SPA dropdown (`GET /api/llm-models`) probes each engine and, when it is up, **replaces the YAML model list with live `/v1/models`**.

## 1. Pick an engine (do not use port 8000)

BankAI FastAPI already binds **`:8000`**. Run vLLM on **8001**.

| Engine | Repo | Default API | BankAI provider id |
|--------|------|-------------|--------------------|
| vLLM | [docker-vllm-serving-engine](https://github.com/johnsonhk88/docker-vllm-serving-engine) | `http://127.0.0.1:8001/v1` | `vllm` |
| SGLang | [sglang-llm-inference-engine](https://github.com/johnsonhk88/sglang-llm-inference-engine) | `http://127.0.0.1:30000/v1` | `sglang` |
| Strata | [strata-llm-inference-engine](https://github.com/johnsonhk88/strata-llm-inference-engine) | `http://127.0.0.1:8080/v1` | `strata` |
| LM Studio | — | `http://127.0.0.1:1234/v1` | `lm-studio` |
| Ollama | — | `http://127.0.0.1:11434` | `ollama` |

Catalog file: `backend/config/llm_providers.yaml` (edit `base_url` if you changed ports).

### vLLM (example)

From the vLLM repo, map host **8001**:

```bash
./src/run-vllm-docker.sh \
  --model Qwen/Qwen3-0.6B \
  --name qwen-small \
  --port 8001 \
  --gpu-memory-utilization 0.2
curl http://localhost:8001/v1/models
```

### SGLang (example)

```bash
./src/run-sglang-docker.sh \
  --model Qwen/Qwen3-0.6B \
  --name qwen-small
curl http://localhost:30000/v1/models
```

The SGLang launcher already adds `--enable-metrics` (`/metrics` for Prometheus).

### Strata (example)

Follow [strata-llm-inference-engine](https://github.com/johnsonhk88/strata-llm-inference-engine) (`./scripts/launch-docker.sh`). When healthy:

```bash
curl -fsS http://127.0.0.1:8080/health
```

If Strata uses `API_KEY`, copy it into `infra/.env` as `STRATA_API_KEY`.

## 2. Use it in BankAI

1. Start BankAI (`cd infra && docker compose up -d`, then `cd frontend && npm run dev`).
2. Open **New analysis**.
3. The **Model** dropdown lists local engines. Unavailable ones show the probe error.
4. When vLLM/SGLang/Strata is up, model IDs refresh from `/v1/models` (every 15s).
5. Submit. The worker calls LiteLLM with that `base_url`.

From a container, `host.docker.internal` reaches engines on the host (Compose already documents this for LM Studio).

## 3. GPU / CPU / RAM (Prometheus + Grafana)

Same pattern as the vLLM/SGLang `monitoring/` stacks: Prometheus scrapes DCGM + node-exporter + engine `/metrics`; Grafana draws the dashboard.

This host often already has Grafana `:3000`, DCGM `:9400`, and node-exporter `:9100`. BankAI monitoring therefore binds Grafana to **`:3001`** and scrapes **existing** exporters on the host. It does not start a second DCGM.

```bash
docker compose -f infra/monitoring/docker-compose.yml up -d
```

| UI | URL |
|----|-----|
| Grafana (admin / admin) | http://localhost:3001 |
| Dashboard **BankAI Local LLM** | http://localhost:3001/d/bankai-local-llm |
| Prometheus | http://localhost:9090 |
| Prometheus targets | http://localhost:9090/targets |

You should see GPU util, VRAM, host CPU, host RAM. Engine jobs (`vllm` `:8001`, `sglang` `:30000`, `strata` `:8080`) stay **DOWN** until that engine is running — that is expected.

If nothing is exporting DCGM/node on this machine:

```bash
docker compose -f infra/monitoring/docker-compose.yml --profile exporters up -d
```

(Only if `:9400` and `:9100` are free.)

The BankAI SPA **GPU monitor** nav link opens the Grafana dashboard.

## 4. Analysis workflow

1. **Source** — upload PDFs and/or tick existing documents.
2. **Engine** — agent + local/cloud model.
3. **Question** — default is a full analysis report; edit if you want.
4. **Run** — Celery runs parse → PII → RAG (Qdrant) → markdown + transactions in Postgres.
5. **History** — **Dashboard** rolls up succeeded transactions by year/quarter. Download Markdown or Print/PDF.
