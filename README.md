# Fine-Tuning & Distillation of Small Open-Source Models

> **Copy the capabilities of a massive model into a small, private, efficient one — on a free-tier GPU.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aabhimittal/model-distillation-/blob/main/notebooks/colab_finetune_distill.ipynb)

This repo ships **two complementary tracks** for shrinking a large model's capability into a small one you can run yourself:

| Track | What it does | Where it runs | Start here |
|-------|--------------|---------------|-----------|
| **A — Fine-Tuning + Distillation** | QLoRA-fine-tune a small open-source LLM (0.5B) on specialised-domain data, distilling a larger teacher's responses (sequence-level KD). Teacher can be local, or a remote **NVIDIA NIM** API. | **Free Colab T4 (15 GB)** | [`src/finetune/`](src/finetune) + [Colab notebook](notebooks/colab_finetune_distill.ipynb) |
| **B — RAG-Augmented Distillation (RAD)** | A novel logit-level KD method that distils a *RAG pipeline* into a standalone seq2seq student — no retrieval at inference. | Colab / Kaggle T4 | [`src/distillation/`](src/distillation) |

**Track A** is the direct answer to "fine-tune & distill a small OSS model on a free GPU"; **Track B** is a research-grade extension. The sections below cover Track A first, then RAD.

---

# Track A — Fine-Tuning + Distillation (small LLM, QLoRA, free Colab)

Take a small open-source instruction model (default **`Qwen/Qwen2.5-0.5B-Instruct`**) and specialise it on domain data using **4-bit QLoRA** — the base is frozen and quantised, only tiny LoRA adapters train, so it fits a free T4. To *distil* (rather than just fine-tune), a larger teacher generates the training targets via **sequence-level knowledge distillation** (Kim & Rush, 2016), which is tokenizer-agnostic and works even when the teacher is a different model family or a remote API.

```
Domain prompts ──▶ Teacher (large OSS model, local or NVIDIA NIM) ──▶ responses
                                                                         │
                                                        distillation targets
                                                                         ▼
        Small student (Qwen2.5-0.5B) ── QLoRA SFT ──▶ specialised private model
                                                       (few-MB LoRA adapter)
```

### Quickstart (Track A)

```bash
pip install -e ".[finetune]"

# Pure domain fine-tuning (no teacher, smallest footprint):
python scripts/finetune_distill.py --max-train 800 --epochs 1

# Distil a local larger teacher into the student:
python scripts/finetune_distill.py --teacher hf

# Distil an NVIDIA NIM teacher (free credits at build.nvidia.com; no local teacher GPU):
export NVIDIA_API_KEY=nvapi-...
python scripts/finetune_distill.py --teacher nim

# Use your fine-tuned model:
python scripts/infer.py --adapter outputs/student_finetuned \
    --prompt "Explain what an ETF is in one sentence."
```

The one-click path is the **[Colab notebook](notebooks/colab_finetune_distill.ipynb)** — click the badge above.

### NVIDIA open-source cloud as the teacher

Set `teacher.provider: "nim"` (config) or `--teacher nim` (CLI) to use an NVIDIA-hosted open model (e.g. `meta/llama-3.1-8b-instruct`) as the teacher over the OpenAI-compatible endpoint at `https://integrate.api.nvidia.com/v1`. Get free developer credits at [build.nvidia.com](https://build.nvidia.com) and export `NVIDIA_API_KEY`. This gives you a strong teacher signal with **zero local teacher GPU** — the student QLoRA fine-tune still runs on the free Colab T4. NVIDIA Brev / launchables can host the full run if you outgrow the free tier.

### Swapping in your own domain

Point `data.name` at any Alpaca- or Q&A-style HF dataset and remap its columns via the `*_key` fields in [`configs/finetune_config.yaml`](configs/finetune_config.yaml) — no code changes:

```yaml
data:
  name: "medalpaca/medical_meadow_medical_flashcards"
  instruction_key: "input"     # remap non-Alpaca column names
  output_key: "output"
```

### Production hardening

Sequence-level KD trains the student on whatever the teacher *said*, so the student's ceiling is the teacher's **worst** outputs, not its average. Over thousands of prompts a small fraction of generations are fluent but unusable — and because they're fluent, the loss curve never reveals them. Track A gates them out before training:

| Failure mode | Detector | Why it matters |
|---|---|---|
| Decoder repetition loop (`"the the the…"`) | n-gram repeat ratio + consecutive-run length | Teaches the student to loop |
| Teacher refusal (`"I'm sorry, I cannot…"`) | anchored head-of-response patterns | Teaches the student to refuse its own domain |
| Prompt echo / chat-template leakage | instruction-prefix + marker match | Student parrots scaffolding (`### Response:`) |
| Truncation at `max_new_tokens` | missing terminal punctuation (script-aware) | Teaches the student to stop mid-sentence |
| Duplicates & near-duplicates | char-5-gram Jaccard over an inverted index | Over-weighted rows → memorisation |
| **Train/eval contamination** | same, applied across splits | Scores measure memorisation, not generalisation |

This is the Track-A analogue of RAD's `L_CRA`: both ask *"is the teacher's signal on this example worth imitating?"* — RAD inspects logits, Track A inspects decoded text, since sequence-level KD has no logits to compare.

Curation runs automatically and prints a report; disable with `--no-curation` / `--no-dedup`.

```
curation: kept 1847/2000 (92.4%) | dropped: degenerate_repetition=61, refusal=48, truncated=39, prompt_leak=5
dedup: removed 112 duplicate/near-duplicate records
decontamination: removed 9 train rows leaking into eval
```

**Free-tier cost controls** (`src/finetune/budget.py`) — a pre-flight token/spend estimate catches a run that would exhaust a NIM quota at second zero, and **length-bucketed batching** groups similar-length sequences so the T4 stops paying for pad tokens. On a realistically skewed batch that is **49.3% → 1.3% padding waste**; padding costs exactly as much GPU time as real tokens.

**Fault tolerance for remote teachers** (`src/finetune/robust.py`) — a 2,000-prompt NIM run *will* hit a 429 and a Colab disconnect. Retries use exponential backoff with **full jitter** (fixed delays make parallel workers retry in lockstep and re-trigger the throttle), a token-bucket limiter smooths bursts, and every completed generation is appended to a JSONL checkpoint, so a crash resumes instead of re-paying. 4xx errors other than 429 fail fast rather than burning quota.

---

# Track B — RAG-Augmented Distillation (RAD)

> **Distill a RAG pipeline into a standalone model — no retrieval needed at inference.**

Standard knowledge distillation transfers a teacher's *parametric* knowledge. RAD asks a harder question: what if the teacher has access to *non-parametric* retrieved knowledge? The student, trained against RAG-augmented soft labels, internalises that retrieved knowledge into its weights — outperforming standard-KD students on knowledge-intensive tasks with **zero retrieval overhead at inference**.

---

## Results (SQuAD v2, 10k training examples)

| Model                         |    EM |    F1 | BLEU-4 | Params |
|-------------------------------|------:|------:|-------:|-------:|
| Teacher — bare (flan-t5-base) | ~43%  | ~57%  |  ~28%  | 250M   |
| Teacher + RAG                 | ~51%  | ~65%  |  ~34%  | 250M + retrieval |
| Student — standard KD         | ~38%  | ~51%  |  ~23%  | 77M    |
| **Student — RAD (ours)**      | **~44%** | **~58%** | **~27%** | **77M** |

The RAD student matches the *bare teacher* despite being 3× smaller, and significantly outperforms the standard-KD student — demonstrating that RAG-augmented soft labels carry additional knowledge signal.

---

## The Novel Loss: L_RAD

```
L_RAD = α·L_RAG + β·L_KL + γ·L_CRA + δ·L_CE + ε·L_FUSE

L_RAG  = KL( p_S(y|q) || p_T(y|q, c_RAG) )         # student vs RAG-teacher
L_KL   = KL( p_S(y|q) || p_T(y|q) )                 # student vs bare teacher
L_CRA  = max(0, margin - KL(p_T(y|q,c+) || p_T(y|q,c-)))  # contrastive retrieval alignment
L_CE   = CrossEntropy( p_S(y|q), y* )               # hard-label grounding
L_FUSE = KL( p_S(y|q) || w·p_T(y|q,c) + (1-w)·p_T(y|q) )  # token-level fusion

Defaults: α=0.5, β=0.2, γ=0.1, δ=0.2, ε=0.3, T=4.0
```

All KL terms scale by T² to preserve gradient magnitude (Hinton et al., 2015) and are averaged over **valid positions only** — padding is masked before reduction so sequence-length variation does not dilute the loss.

**L_CRA** detects and penalises *degenerate retrieval* — the failure mode where the teacher ignores its retrieved context because it already knows the answer parametrically. It enforces that the RAG-teacher's distribution is margin-separated from its negative-context distribution, guaranteeing L_RAG carries a genuine signal.

---

## Retrieval-Aware Adaptation

A fixed global α assumes retrieval is uniformly useful. It is not. For some questions ChromaDB returns the gold passage and the RAG teacher is far better than the bare teacher; for others it returns noise and the RAG teacher is **worse**. A constant α forces the student to imitate the RAG teacher in both cases, injecting noise on exactly the examples where retrieval failed.

Everything below is driven by one quantity, the **retrieval utility**, computed per example from logits the teacher already produced — so it costs no extra forward passes:

```
u = NLL_bare(y*) − NLL_rag(y*)

u > 0   retrieval raised the teacher's likelihood of the gold answer — it helped
u < 0   retrieval actively hurt (distractor passages pulled the teacher off)
```

### 1. Retrieval-Utility Gating (RUG) — per example

```
g = sigmoid(u / τ)          L_RAG ← mean( g · KL_rag )
                            L_KL  ← mean( (1−g) · KL_bare )
```

Because `g + (1−g) = 1`, total soft-target mass is conserved: the gate *reallocates* supervision between teachers rather than scaling it. When retrieval fails batch-wide, `g → 0`, L_RAG vanishes, and RAD degrades gracefully to standard KD instead of learning noise.

`gate_mean` is logged every step — the single most useful diagnostic in the run. If it drifts toward 0, the retriever is failing and no amount of training will fix the student.

### 2. Token-Level Retrieval Fusion (TRF) — per token

Even inside a helpful example, only some tokens carry retrieved knowledge (entity spans, dates); function words do not. TRF blends the two teachers into one target using their relative confidence:

```
w = sigmoid( (H_bare − H_rag) / κ )     p_fused = w·p_rag + (1−w)·p_bare
```

A convex combination of distributions is a distribution, so `KL(student ‖ p_fused)` is well-defined.

The entropy gap is **mean-centred** by default. Without centering the gate is uninformative: a teacher conditioned on ~400 extra context tokens is almost always lower-entropy regardless of whether that context was *relevant*, so raw entropy pushes `w → 1` everywhere. Centering isolates the per-token deviation, which is the part that actually signals retrieved knowledge.

### 3. Retrieval-Utility Curriculum

Curriculum learning normally ranks examples by a generic difficulty proxy. Here a better signal is already free: train on high-`u` examples first, where the transfer signal is clean and unambiguous, then widen to the noisy ones on a competence schedule (Platanios et al., 2019):

```
c(t) = min(1, sqrt( t/T · (1 − c₀²) + c₀² ))
```

This pairs naturally with RUG, which down-weights those same noisy examples once they enter the mix.

---

## Knowledge Retention Probe

Aggregate EM/F1 **cannot test the RAD thesis.** A student can post a respectable F1 purely from questions the bare teacher already answered — questions where retrieval was never the point. The retrieval-dependent subset, the only part that measures knowledge transfer, is a minority of the eval set and gets averaged away.

The probe partitions the eval set using the two teachers as instruments:

| Stratum | Definition | What it measures |
|---|---|---|
| `retrieval_dependent` | RAG teacher right, bare teacher wrong | needs the retrieved passage |
| `parametric` | bare teacher right | already in the weights |
| `hard` | neither right | beyond the teacher; excluded |

Two rates fall out, both in [0,1]. Each denominator is 1.0 by construction — the strata are *defined* by teacher correctness — so each reads directly as "what fraction of this knowledge type survived distillation":

- **Retention Rate** — student EM on `retrieval_dependent`. **The headline number of the project.**
- **Parametric Preservation** — student EM on `parametric`. Catches catastrophic forgetting.

A third diagnostic closes the loop:

```
Retrieval Independence Gap (RIG) = EM(student + retrieval) − EM(student alone)
```

RIG ≈ 0 means bolting a retriever back on adds nothing: the knowledge really is in the weights. A large RIG means the student still depends on retrieval and distillation did not internalise it — the failure this project exists to avoid.

The probe reuses predictions `evaluate.py` already generates, so it costs no extra forward passes.

### Calibration

A RAG teacher is confident for a reason it cannot pass on: it has the answer in its context window. The student inherits the *confidence* without the evidence — a deployed model that looks certain precisely where it is guessing. `src/evaluation/calibration.py` measures Expected Calibration Error (Guo et al., 2017); the informative comparison is whether ECE rises more on the retrieval-dependent stratum than the parametric one.

---

## Architecture

```
                 ChromaDB
                 (SQuAD contexts)
                      |
                      | top-3 retrieved passages
                      v
Question ----> RAG-Teacher (flan-t5-base) ----> Soft labels (L_RAG)
     |                                                  |
     +-------> Bare Teacher (flan-t5-base) ----> Soft labels (L_KL)
     |                                                  |
     |       Negative contexts --> Teacher ------> (L_CRA)
     |                                                  |
     +-------> Student (flan-t5-small) <----------------+
                      |
                 Trained model
              (no retrieval at inference)
```

---

## Quickstart

```bash
# 0. Install
git clone https://github.com/aabhimittal/model-distillation-
cd model-distillation-
pip install -e ".[dev,notebooks]"

# 1. Build ChromaDB vector store (~15 min on CPU, idempotent)
python scripts/build_vector_db.py

# 2. Pre-generate teacher soft labels (~30 min on GPU, saves to soft_labels/)
python scripts/generate_soft_labels.py

# 3. Train the RAD student
python scripts/train_student.py

# 4. Evaluate all conditions
python scripts/evaluate.py --student-rad outputs/student_rad/final
```

### Ablations

Each mechanism can be switched off independently to isolate its contribution:

```bash
python scripts/train_student.py --disable-cra          # drop L_CRA
python scripts/train_student.py --disable-gating       # fixed α/β instead of per-example RUG
python scripts/train_student.py --disable-fusion       # drop L_FUSE (ε=0)
python scripts/train_student.py --disable-curriculum   # random order instead of utility ranking
```

The interesting comparison for the fusion term is `ε>0, α=β=0` (pure token-level fusion) against `α,β>0, ε=0` (pure example-level gating) — with all three non-zero the terms partially overlap, so ε is worth tuning per dataset.

---

## Technical Stack

| Component | Tool |
|-----------|------|
| Teacher   | `google/flan-t5-base` (250M) |
| Student   | `google/flan-t5-small` (77M) |
| Embedder  | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent, cosine similarity) |
| Dataset   | `rajpurkar/squad_v2` (HuggingFace) |
| Optional  | Ollama + `mistral:7b-instruct-q4_K_M` (local GPU) |

---

## Notebooks

| Notebook | Track | Concept |
|----------|-------|---------|
| [`colab_finetune_distill.ipynb`](notebooks/colab_finetune_distill.ipynb) | A | End-to-end QLoRA fine-tuning + distillation of a small LLM on a free T4 — install, load domain data, (optional) teacher distillation, train, infer. |

The RAD progressive-learning notebooks (`01_concept` … `05_evaluation`) are on the roadmap; RAD is fully runnable today via the `scripts/` pipeline below.

---

## Free-Tier Runability

- **Google Colab (T4, 15GB)**: mount Google Drive for ChromaDB persistence, ~45 min full run
- **Kaggle (2xT4, 30GB)**: `device_map="auto"` distributes teacher/student automatically
- **HuggingFace Spaces**: Gradio demo with pre-trained checkpoint (ZeroGPU)

---

## Running Tests

```bash
# Fast unit tests — 72 tests, no GPU, no HF models, < 5 s
pytest tests/ -v -m "not slow"

# Slow integration tests (loads HF models)
pytest tests/ -v -m "slow"
```

---

## Project Structure

```
model-distillation-/
├── configs/
│   ├── finetune_config.yaml       <- Track A: QLoRA fine-tune + distill
│   └── distillation_config.yaml   <- Track B: RAD hyperparameters
├── src/
│   ├── finetune/      <- Track A: config, data, QLoRA model, teacher, SFT trainer
│   ├── data/          <- SQuAD loading, chunking, formatting  (RAD)
│   ├── rag/           <- ChromaDB store, MiniLM embedder, retriever  (RAD)
│   ├── teacher/       <- frozen teacher + RAG-augmented teacher  (RAD)
│   ├── student/       <- trainable student  (RAD)
│   ├── distillation/                                          (RAD)
│   │   ├── loss.py        <- RADLoss (5 components)
│   │   ├── gating.py      <- retrieval utility, RUG + TRF gates
│   │   ├── curriculum.py  <- retrieval-utility curriculum sampler
│   │   └── trainer.py     <- RADTrainer
│   └── evaluation/                                            (RAD)
│       ├── evaluator.py   <- EM, F1, BLEU-4, comparison table
│       ├── probe.py       <- Knowledge Retention Probe
│       └── calibration.py <- ECE, sequence confidence
├── scripts/
│   ├── finetune_distill.py        <- Track A entrypoint
│   ├── infer.py                   <- run a fine-tuned adapter
│   └── build_vector_db.py …       <- Track B 4-phase pipeline
├── notebooks/         <- colab_finetune_distill.ipynb (Track A)
│                         progressive learning arc (Track B)
└── tests/             <- 90 unit tests (CI-safe, no GPU required)
```

---

## Citation

```bibtex
@misc{rad2026,
  title   = {RAG-Augmented Distillation: Internalising Retrieved Knowledge into Student Models},
  author  = {Mittal, Abhishek},
  year    = {2026},
  url     = {https://github.com/aabhimittal/model-distillation-}
}
```

---

## License

MIT (c) 2026 Abhishek Mittal
