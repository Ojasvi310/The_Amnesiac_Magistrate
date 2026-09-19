# Continual Counsel

An edge-deployable legal/regulatory compliance assistant with continual learning — quarterly regulatory updates via O-LoRA fine-tuning, TIES-merging, and local quantized inference.

This project demonstrates how to adapt a localized language model to sequential, disparate legal regimes (Q1, Q2, etc.) **without catastrophic forgetting** and **without retaining original confidential texts**.

## Core Architecture: Overcoming Catastrophic Forgetting

Traditional fine-tuning methods suffer from catastrophic amnesia because sequential updates overwrite the vector subspaces used by earlier tasks. 

Our solution leverages **Orthogonal Low-Rank Adaptation (O-LoRA)**:
1. **Mathematical Isolation:** After training on a legal regime (e.g., Q1), we extract the principal components of the model's intermediate activations and save an orthonormal basis checkpoint.
2. **Orthogonal Projection:** When training on the next regime (e.g., Q2), gradient updates are explicitly projected *away* from the previous basis. This guarantees that new knowledge is written into an orthogonal subspace, preserving foundational precedents without interference.
3. **No Data Replay:** Because previous knowledge is protected mathematically via the basis checkpoint, there is zero need to store highly confidential historical training corpora in the edge vaults.
4. **Edge-Hardware Viable:** The core model remains entirely frozen and quantized to 4-bit (GGUF format), ensuring the entire stack runs flawlessly on constrained edge hardware.

## Quick Start (Running Locally)

To test the offline edge dashboard and inference engine on your own machine:

1. **Start the FastAPI Inference Server:**
   ```bash
   # Windows:
   .\run_api.bat
   ```

2. **Start the Dashboard:**
   In a separate terminal, launch the Streamlit interface:
   ```bash
   streamlit run dashboard/app.py
   ```
   Navigate to `http://localhost:8501`. You can test the interference resilience by asking questions from Regime 1 (e.g., data breach timelines) and Regime 2 (e.g., AI incident timelines).

---

**Execution Environments**: Two physically separate execution environments:

| Loop | Machine | Key tools |
|---|---|---|
| **Offline (training)** | Google Colab (GPU) | peft, transformers, bitsandbytes, llama.cpp |
| **Online (inference)** | Local CPU | llama-cpp-python, faiss-cpu, FastAPI, Streamlit |

The boundary between them is enforced at the import-graph level (not just by convention)
and verified by an automated smoke test that runs the inference container with
`--network none`. The export/handoff protocol between machines is a first-class pipeline
stage, not an afterthought.

---

## Quick start: full toy cycle across both machines

This walkthrough reproduces the complete two-machine flow using the synthetic toy
regulations included in the repo (`regime_q1` = Regulation A, `regime_q2` = Regulation B).
No real regulatory data, no gated model checkpoints.

### Prerequisites (local machine)

```bash
# Python 3.11, Docker, git
pip install -r requirements-online.txt
```

### Step 1 — Colab: clone at a pinned commit

Open `notebooks/colab_train.ipynb` in Google Colab with a GPU runtime (T4 is sufficient
for the proxy profile).

In Cell 1, set:
```python
REPO_URL = "https://github.com/YOUR_ORG/continual-counsel.git"
COMMIT_OR_TAG = "<your-commit-sha>"  # pin explicitly; recorded in export bundle manifest
REGIME_NAME = "regime_q1"
```

Run Cell 1 to clone the repo at that commit. The commit hash is logged and will appear
in every export bundle manifest and adapter registry entry produced by this session.

### Step 2 — Colab: install offline deps

Cell 2: `pip install -r requirements-offline.txt`

This never touches `requirements-online.txt`. bitsandbytes requires a GPU-visible torch
build and would break pip resolution on the local machine.

### Step 3 — Colab: supply the input bundle (toy run)

For the toy demo, the synthetic regulatory docs are already in the cloned repo under
`data/regimes/regime_q1/`. Set `INPUT_BUNDLE_DIR = None` in Cell 3 and the pipeline
will use the repo-local data directly.

For a real run: mount Google Drive (or pull from wherever your input bundle lives)
and set `INPUT_BUNDLE_DIR` to the path containing:
- `data/regimes/<regime>/` — new regulatory docs
- `faiss_snapshot/` — current FAISS index (built locally, see Step 7)
- `adapter_stack/` — current adapter + basis checkpoint

### Step 4 — Colab: run the pipeline

Cell 4 runs:
```bash
bash scripts/run_quarter_update_colab.sh regime_q1
```

This invokes `src/offline/pipeline.py`, which executes Steps A→F:
- **A**: Golden replay capture (with self-consistency + grounding filters)
- **B**: Compose training batch (70% new regime + 30% replay)
- **C**: O-LoRA training with orthogonality penalty
- **D**: Benchmark gate (BWT ≤ 3% degradation)
- **E**: TIES-merge every 2 quarters (configurable)
- **F**: GGUF export (Q4_K_M by default)

GPU detection is automatic — the script logs the GPU name and VRAM at startup and
adjusts fp16/bf16 accordingly. The proxy profile (`Qwen2.5-1.5B-Instruct`) runs
comfortably on a T4.

### Step 5 — Colab: assemble and download the export bundle

Cell 5 runs `scripts/assemble_export_bundle.sh`, which zips the bundle and writes the
manifest. Download from the Colab Files panel (or copy to Drive if you configured a
Drive destination).

The export bundle at `exports/regime_q1_<timestamp>.tar.gz` contains:
- `adapter/` — PEFT adapter files
- `model.gguf` — quantized GGUF (Q4_K_M)
- `basis_checkpoint.npz` — orthonormal basis state
- `golden/` — golden replay set
- `benchmark_results.json` — per-quarter benchmark scores
- `manifest.json` — SHA-256 per file + commit hash + Python version
- `manifest.sha256` — hash of the manifest itself

### Step 6 — Local: verify the export bundle

```bash
mkdir -p exports
tar -xzf regime_q1_<timestamp>.tar.gz -C exports/
bash scripts/verify_export_bundle.sh exports/regime_q1_<timestamp>/
```

`verify_export_bundle.sh` recomputes SHA-256 for every file, checks sizes, verifies
`manifest.sha256`, and only then calls `src/audit/registry.py` to register the adapter.
If any file is missing, truncated, or tampered, the bundle is rejected and the adapter
is not registered. This is the structural equivalent of Step D's benchmark gate.

> **Why this matters**: A silent partial export — inference service running on a stale
> adapter because one file didn't copy over — is a worse failure mode than a network
> call would be, because it fails quietly. The manifest verification gate prevents this.

### Step 7 — Local: build the FAISS retrieval index

```bash
python -m src.online.retrieval build --docs-root data/regimes/ --output-dir data/faiss_index/
```

This is cheap on CPU and builds the index from the same docs that are in the repo.
The index hash (SHA-256 of `index.faiss`) is recorded in every audit log entry,
so "what the model could retrieve at query time" is pinned and auditable.

### Step 8 — Local: start the inference API

```bash
CC_GGUF_PATH=exports/regime_q1_<timestamp>/model.gguf \
CC_ADAPTER_DIR=exports/regime_q1_<timestamp>/adapter/ \
CC_INDEX_DIR=data/faiss_index/ \
uvicorn src.online.api:app --host 0.0.0.0 --port 8000
```

Or via Docker (recommended — proves the no-network guarantee):
```bash
docker build -t continual-counsel .
docker run --network none -p 8000:8000 \
  -v $(pwd)/exports/regime_q1_<timestamp>:/data \
  -e CC_GGUF_PATH=/data/model.gguf \
  -e CC_ADAPTER_DIR=/data/adapter/ \
  -e CC_INDEX_DIR=/data/faiss_index/ \
  continual-counsel
```

`--network none` blocks all outbound calls from the container. The inference service
still answers queries via the mapped port because `--network none` disables external
routing, not loopback.

### Step 9 — Local: query the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the deadline for breach notification under Regulation A?"}'
```

Example response:
```json
{
  "query_id": "uuid-...",
  "answer": "Under Section 4.1 of Regulation A, a data controller must notify...",
  "status": "ok",
  "retrieved_regime_tags": ["regime_q1"],
  "index_version": "sha256:abc...",
  "adapter_version": "sha256:def...",
  "verified": true,
  "flagged_claims": [],
  "timestamp": "2024-01-15T10:23:45"
}
```

If retrieval confidence is too low, you get an escalation response instead of a
hallucinated answer:
```json
{"status": "escalate", "message": "Insufficient grounding — escalate to human review"}
```

### Step 10 — Local: open the dashboard

```bash
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501`. The dashboard shows:
- Accuracy/BWT/FWT curves over quarters
- Adapter footprint over time (showing the merge step bounding growth)
- Basis rank sawtooth plot (growth → reset at merge)
- Hallucination rate by quarter
- Audit log with full per-query lineage: adapter → merge parents → export bundle → repo commit

### Step 11 — Local: run the test suite

```bash
pytest -m "not colab_only" -v tests/
```

This runs all local tests. GPU/Colab-only tests are skipped automatically.
See [CI](#ci) for what runs in GitHub Actions.

---

## DPO re-validation mode

This implementation uses **option (a)**: after a DPO polishing pass, the Step D
benchmark gate is re-run on the post-DPO adapter. If BWT degrades past the 3%
threshold, the pre-DPO adapter state (saved before the DPO pass begins) is restored
and the DPO pass is rejected. The pipeline proceeds with the pre-DPO adapter.

**Why option (a) and not option (b)** (folding the orthogonality penalty into the DPO
objective directly): option (a) gives a cleaner audit trail — the accepted adapter
either passed the benchmark gate after DPO or it didn't, and "adapter X was rejected
at DPO re-validation" is a legible fact in the run log. Option (b) requires correctly
tuning a custom DPO loss that combines KL, preference, and orthogonality terms; the
interaction between those terms on a small proxy model is poorly characterized, and the
"DPO polished with orthogonality-aware objective" claim would be hard to verify without
an ablation study that the spec doesn't require.

DPO is disabled by default. Enable in `configs/training.yaml`:
```yaml
dpo:
  enabled: true
  revalidation_mode: "gate"  # option (a)
```

---

## proxy vs full_scale profile

| | proxy (default) | full_scale |
|---|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` | `meta-llama/Llama-3.1-8B-Instruct` (fallback: `mistralai/Mistral-7B-Instruct-v0.3`) |
| LoRA rank ceiling | 32 | 64 |
| VRAM budget | auto-detect at runtime | 40 GB (A100); L4 may need rank cap |
| GGUF quant level | Q4_K_M | Q5_K_M (ablation: Q8_0) |
| Colab tier needed | T4 (15 GB VRAM) | A100 recommended |
| HF token required | No | Yes (Llama 3.1 is gated) |

To switch to full_scale:
```bash
bash scripts/run_quarter_update_colab.sh regime_q1 --profile full_scale
```

The full_scale profile is not exercised by default and exists to document exactly what
changes — not to assert that 8B-scale results would be better without running them.

---

## Repository layout

```
continual-counsel/
  configs/
    base.yaml              # model profiles, paths, retrieval settings
    training.yaml          # all hyperparameters from §4 table
  requirements-offline.txt # Colab/GPU deps (torch, peft, transformers, bitsandbytes)
  requirements-online.txt  # local/CPU deps (faiss-cpu, llama-cpp-python, fastapi)
  notebooks/
    colab_train.ipynb      # 5 bootstrap cells only; no pipeline logic
  data/
    regimes/
      regime_q1/           # Regulation A (synthetic: data privacy)
      regime_q2/           # Regulation B (synthetic: AI accountability)
    replay_cache/          # golden + refreshed replay sets
  src/
    offline/               # GPU only — training, merging, GGUF export
      replay_gen.py        # Step A: golden capture + filters
      train_adapter.py     # Step B+C: O-LoRA + orthonormal basis
      distill.py           # feature-level distillation loss
      dpo_polish.py        # DPO pass + option (a) re-validation
      validate.py          # Step D: benchmark gate
      merge.py             # Step E: TIES-merge (from scratch)
      export.py            # Step F: GGUF conversion + bundle assembly
      pipeline.py          # orchestrates A→F
    online/                # CPU only — ZERO training deps importable
      retrieval.py         # FAISS index build/query + versioning
      confidence_gate.py   # retrieval confidence gate (raises, not just logs)
      verify.py            # self-verification pass
      infer.py             # full inference pipeline
      api.py               # FastAPI entrypoint
    audit/
      registry.py          # adapter version registry (SQLite)
      log.py               # tamper-evident audit log (hash-chained)
    eval/
      benchmarks.py        # per-quarter benchmark construction
      confusion_set.py     # cross-regime adversarial eval
      metrics.py           # BWT, FWT, footprint, basis rank, hallucination rate
      report.py            # curve data for dashboard
  dashboard/
    app.py                 # Streamlit, local only, never imports src/offline/
  exports/
    <regime>_<timestamp>/  # one per Colab export bundle
  scripts/
    run_quarter_update_colab.sh   # invoked inside Colab; calls pipeline.py
    assemble_export_bundle.sh     # zips bundle, writes manifest (Colab side)
    verify_export_bundle.sh       # verifies bundle on local receipt
    smoke_test_no_network.sh      # Docker + --network none smoke test
    run_colab_tests.sh            # runs colab_only pytest subset in Colab
    run_full_eval.sh
  Dockerfile                      # local inference service only
  tests/
    conftest.py
    test_basis.py
    test_ties_merge.py
    test_confidence_gate.py
    test_grounding_filter.py
    test_dpo_revalidation.py
    test_export_bundle.py
```

---

## Hyperparameters (all config-driven)

See `configs/training.yaml` for the full table with sweep ranges. Nothing is hardcoded
in source files except one intentional rough edge: `batch_size = 4` in
`src/offline/train_adapter.py` (marked with a TODO for per-GPU dynamic sizing once
VRAM detection is wired in fully).

---

## Security and audit

Every query writes an immutable, hash-chained record to the SQLite audit log:
query text, retrieved chunk IDs and regime tags, retrieval index hash active at query
time, adapter version hash, confidence scores, verification verdict, final answer, and
timestamp. Each record's hash includes the previous record's hash — tampering with any
record is detectable by `AuditLog.verify_chain()`.

Every registered adapter is traceable back to:
- The exact export bundle that delivered it (manifest hash)
- The repo commit the offline training code was checked out at (from the manifest)
- The benchmark scores it passed at acceptance time
- Its merge lineage (which prior adapters were merged into it)

The dashboard exposes this lineage for any past query.
