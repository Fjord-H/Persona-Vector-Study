# Persona Vectors Research Dashboard

> ## ⚠ Deprecated — displays pre-correction results
>
> These dashboards show the accuracy figures from before the August 2026 audit
> (92.5%, 64-74%, etc.). Those figures are retracted — see the [main
> README](../README.md#known-issues-august-2026-audit) and `defect_report.md`.
>
> The live URLs below are also dead; the EC2 instances were terminated when the
> figures were retracted. Kept here as a historical/portfolio artifact only —
> do not present these numbers as current results.

Interactive dashboards for visualizing the research journey, built with Streamlit.

## Two Versions

### Dashboard v0 (Discovery Process)
**Port 8501** — the original tone-vs-content discovery and the (at-the-time
believed) overfitting reality check.

### Dashboard v1 (Complete Research)
**Port 8502** — cross-model results across GPT-2, Qwen, and Llama, plus layer
analysis and the failed-experiments log (weighting, ensembling, per-category
vectors).

---

## Installation
```bash
pip install -r requirements.txt
```

## Local usage
Run from the repo root so relative `data/` and `figures/` paths resolve:
```bash
streamlit run dashboard/dashboard_v0.py --server.port 8501
streamlit run dashboard/dashboard_v1.py --server.port 8502
```
- v0: `http://localhost:8501`
- v1: `http://localhost:8502`

## Docker deployment

See `DOCKER.md`. Build from the **repo root** (not from inside `dashboard/`),
since the images need `data/` and `figures/` from outside this folder:
```bash
docker build -f dashboard/Dockerfile.v0 -t persona-vectors-dashboard:v0 .
docker build -f dashboard/Dockerfile.v1 -t persona-vectors-dashboard:v1 .
```

---

## Technology stack

- **Framework:** Streamlit
- **Visualization:** Matplotlib, Seaborn
- **Data:** Pandas, NumPy
- **Deployment:** Docker (EC2 deployment retired — see deprecation notice above)

## Notes

- Dashboards use pre-computed results from the v1 research notebooks.
- Interactive demo panels use simulated scores based on model behavior, not
  live inference.
- Numbers shown throughout predate the August 2026 correction. Treat as a UI/
  deployment artifact, not a results source.
