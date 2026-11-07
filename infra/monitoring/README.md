# BankAI monitoring

Prometheus + Grafana, same idea as the vLLM and SGLang engine repos.

```bash
docker compose -f infra/monitoring/docker-compose.yml up -d
```

- Grafana: http://localhost:3001 (admin / admin) — port **3001** so it does not clash with an existing Grafana on `:3000`
- Dashboard: http://localhost:3001/d/bankai-local-llm
- Prometheus: http://localhost:9090/targets

Scrapes `host.docker.internal:9400` (DCGM), `:9100` (node-exporter), plus vLLM `:8001/metrics`, SGLang `:30000/metrics`, Strata `:8080/metrics`.

To start DCGM + node-exporter here (only if those ports are free):

```bash
docker compose -f infra/monitoring/docker-compose.yml --profile exporters up -d
```
