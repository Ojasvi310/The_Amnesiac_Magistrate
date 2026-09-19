# Continual Counsel

An edge-deployable legal/regulatory compliance copilot that learns sequential regulatory regimes (Q1 → Q2 → Q3 → Q4) **without catastrophic forgetting** and **without retaining original confidential documents** on the edge device.

---

## Architecture

### The Core Problem: Catastrophic Forgetting

When a standard language model is fine-tuned on new legal regulations (e.g., Quarter 2 AI Accountability laws), its gradient updates blindly overwrite the same neural weight subspaces it used to learn Quarter 1 Data Privacy laws. By the time the model has trained on Q4, it has largely forgotten what Q1 said. This is Catastrophic Forgetting — and it is fatal for a legal compliance tool where every precedent matters.

### Solution: Orthogonal Low-Rank Adaptation (O-LoRA)

Instead of overwriting the model's existing knowledge, Continual Counsel learns each new regime into a mathematically separate, protected subspace of the model's weight space.

**How it works, step by step:**

**1. Basis Capture (after Q1 training)**
After training on Q1, the pipeline extracts the principal components of all LoRA A-matrices (the low-rank update matrices). These are stacked and decomposed via QR factorisation to produce an orthonormal basis — a compact mathematical fingerprint of exactly which directions in weight space Q1 "occupies." This is saved as `basis_checkpoint.pt`.

**2. Orthogonal Projection (during Q2 training)**
When training on Q2, every gradient step includes an explicit **Orthogonality Loss**:
```
L_total = L_cross_entropy + λ · L_orthogonality
```
`L_orthogonality` is the squared dot product between the new Q2 LoRA A-matrices and the Q1 basis vectors. Minimising this forces the Q2 updates into the **null space** of Q1 — a completely orthogonal (empty) dimension of the model's brain. Q1's weights are never touched.

**3. Sequential Chaining (Q3, Q4, ...)**
Each new quarter accumulates the previous quarters' bases. Q3 trains orthogonally to the combined Q1+Q2 basis. The basis grows with each quarter, acting as a permanent mathematical "do not touch" fence around all prior knowledge.

**4. TIES-Merging (every N quarters)**
Once every configurable number of quarters, all individual LoRA adapters are merged into a single, unified adapter using **TIES-Merge** (Trim, Elect Sign, Disjoint Average), preventing unbounded adapter growth while preserving all accumulated knowledge.

**5. GGUF Quantization (edge deployment)**
The merged adapter weights are baked into the frozen base model and quantized to 4-bit (Q4_K_M GGUF format) using `llama.cpp`. The resulting binary runs entirely on CPU with no GPU, no internet connection, and no training libraries — making it viable for air-gapped edge deployments.

---

### RAG Pipeline (Online Inference)

At query time, the system runs a multi-stage Retrieval-Augmented Generation pipeline entirely on-device:

```
User Query
    │
    ▼
Sentence Embedding (all-MiniLM-L6-v2, local)
    │
    ▼
FAISS Vector Search (top-k chunks across all regimes)
    │
    ▼
Confidence Gate ──► Escalate to human if max score < threshold
    │
    ▼
ChatML Prompt Assembly (system + context + question)
    │
    ▼
Qwen2.5-1.5B-Instruct GGUF (llama.cpp, CPU-only)
    │
    ▼
Self-Verification Pass (keyword grounding check against retrieved chunks)
    │
    ▼
Audit Log Entry (hash-chained SQLite record)
    │
    ▼
Structured JSON Response
```

Every query is immutably recorded with the exact chunk IDs retrieved, the FAISS index version hash, the adapter version hash, and the final verification verdict. This makes every answer traceable, auditable, and tamper-evident.

---

### Repository Layout

```
continual-counsel/
├── configs/
│   ├── base.yaml              # Model profiles, paths, retrieval settings
│   └── training.yaml          # All training hyperparameters
├── data/
│   └── regimes/
│       ├── regime_q1/         # Regulation A — Data Privacy
│       ├── regime_q2/         # Regulation B — AI Accountability
│       ├── regime_q3/         # Regulation C — Antitrust & Offshore Tax
│       └── regime_q4/         # Regulation D — Cross-Border Tariffs
├── src/
│   ├── offline/               # GPU training pipeline (Colab only)
│   │   ├── pipeline.py        # Orchestrates Steps A → F
│   │   ├── train_adapter.py   # O-LoRA training + orthonormal basis tracking
│   │   ├── validate.py        # BWT benchmark gate
│   │   ├── merge.py           # TIES-merge (implemented from scratch, no mergekit)
│   │   └── export.py          # GGUF conversion + export bundle assembly
│   ├── online/                # CPU inference (zero training library imports)
│   │   ├── retrieval.py       # FAISS index build and query
│   │   ├── confidence_gate.py # Blocks generation if retrieval score too low
│   │   ├── infer.py           # Full RAG + ChatML generation pipeline
│   │   ├── verify.py          # Self-verification against retrieved chunks
│   │   └── api.py             # FastAPI entrypoint
│   ├── audit/
│   │   ├── registry.py        # Adapter version registry (SQLite)
│   │   └── log.py             # Hash-chained immutable audit log
│   └── eval/
│       ├── metrics.py         # BWT, FWT, footprint, hallucination rate
│       └── report.py          # Report generation for dashboard
├── dashboard/
│   └── app.py                 # Streamlit compliance dashboard
├── exports/                   # Trained adapter bundles (one per quarter)
├── runs/                      # Training run logs and validation scores
├── run_api.py                 # Cross-platform API launcher
├── run_local_eval.py          # Benchmark evaluation against local API
└── run_adversarial_test.py    # Cross-regime confusion scoring
```

---

### Training Pipeline (Offline — Runs on Google Colab GPU)

The full offline training pipeline (`src/offline/pipeline.py`) executes these stages sequentially for each new regulatory quarter:

| Step | Module | What it does |
|---|---|---|
| **A** | `replay_gen.py` | Captures golden Q&A pairs via two-seed self-consistency sampling |
| **B+C** | `train_adapter.py` | O-LoRA fine-tuning with orthogonality loss against prior basis |
| **D** | `validate.py` | BWT benchmark gate — rejects adapter if forgetting exceeds 3% |
| **E** | `merge.py` | TIES-merge every N quarters to keep adapter stack bounded |
| **F** | `export.py` | GGUF quantization + export bundle assembly |

The basis checkpoint (`basis_checkpoint.pt`) is the single most important artefact — it is what makes the zero-forgetting guarantee possible.

---

## Running Locally

### Prerequisites

```bash
pip install -r requirements-online.txt
```

### 1. Start the Inference API

```bash
python run_api.py
```

This starts the FastAPI server at `http://localhost:8000` with the Q4 adapter loaded. All inference runs locally — no network calls, no GPU required.

To verify the API is up:
```bash
curl http://localhost:8000/health
```

To query it directly:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the reporting deadline for a data breach under Regulation A?"}'
```

### 2. Start the Dashboard

In a **separate terminal**:
```bash
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501`.

The dashboard has five tabs:
- **Query Assistant** — Chat with the compliance copilot
- **Metrics** — Regime accuracies, BWT, FWT, hallucination rate
- **Confusion Set** — Cross-regime adversarial test scores

### 3. Generate Real Benchmark Scores (Optional)

With the API running, evaluate the model against all four regime benchmark sets:
```bash
python run_local_eval.py
```

This fires the held-out benchmark questions at the running API, scores responses using keyword matching, and writes `report.json` files into `runs/` for the dashboard to display.



## Key Design Decisions

**Zero training imports in `src/online/`**
The entire inference stack imports only `llama-cpp-python`, `faiss-cpu`, `sentence-transformers`, `fastapi`, and standard library. `torch`, `peft`, `transformers`, and `bitsandbytes` are never importable in the inference container. This is enforced structurally — not just by convention.

**Confidence Gate raises, not warns**
If the FAISS retrieval score for a query is below the threshold, `InsufficientGrounding` is raised and generation is blocked entirely. The API returns an escalation response. The model never hallucinates when it has nothing reliable to retrieve from.

