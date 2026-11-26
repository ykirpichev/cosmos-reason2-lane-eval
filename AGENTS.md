# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Single Python project: **Cosmos-Reason2 lane-behavior eval** (BATON-Sample clips, vLLM inference, Streamlit UIs). See `README.md` for the full pipeline.

### System prerequisites (one-time on fresh Ubuntu VMs)

`python3 -m venv` requires the distro venv package (not in the default Cloud Agent image):

```bash
sudo apt-get install -y python3.12-venv
```

Docker and an NVIDIA GPU are only needed for `scripts/serve_vllm.sh` (Cosmos-Reason2-32B inference). The Streamlit apps and offline analysis work without GPU/Docker.

### Python environment

From repo root:

```bash
test -d .venv || python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Optional: `cp .env.example .env` and set `LANE_CACHE_DIR` to a persistent volume.

### Data paths (`scripts/config.py`)

- Committed sample artifacts live under top-level `results/` (legacy layout): `summary.json`, `logs/`, etc.
- Clip videos and `manifest_all.json` are **not** in git. Without `clips/manifest_all.json` (or `<cache>/clips/manifest_all.json`), Streamlit shows “No manifest” until you run `scripts/ingest_baton.py`.
- `LANE_CACHE_DIR` defaults to `./cache`; if `results/` or `clips/` exist at repo root, those legacy dirs are used instead of the cache equivalents.

### Running services (manual — not in the VM update script)

| Service | Command | Notes |
|---------|---------|--------|
| Dashboard | `.venv/bin/streamlit run apps/view_examples.py` | Port 8501 |
| Human labeling | `.venv/bin/streamlit run apps/label_clips.py` | |
| Disagreement review | `.venv/bin/streamlit run apps/review_disagreements.py --server.port 8503` | |
| vLLM (Cosmos) | `scripts/serve_vllm.sh` | Docker + GPU; health: `curl -sf http://127.0.0.1:8000/v1/models` |
| Batch inference | `.venv/bin/python scripts/run_batch.py` | Requires vLLM up |

Use **tmux** for long-running Streamlit/vLLM processes in Cloud Agents.

### Lint / test

No configured linter or test suite. Sanity check:

```bash
.venv/bin/python -m compileall scripts apps
```

### Full pipeline (when GPU + HF access available)

Documented in `README.md`: `ingest_baton.py` → `remap_pseudo_3class.py` → `serve_vllm.sh` → `run_batch.py` → Streamlit apps. Set `HF_TOKEN` or `huggingface-cli login` for BATON-Sample download.
