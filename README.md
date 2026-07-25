# RAG-Augmented Distillation (RAD)

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

## Progressive Learning Notebooks

| Notebook | Concept |
|----------|---------|
| `01_concept_introduction.ipynb` | What is distillation? Why does retrieval matter? |
| `02_rag_setup.ipynb` | Build ChromaDB, visualise retrieved passages with t-SNE |
| `03_teacher_soft_labels.ipynb` | Compare teacher distributions with/without RAG (KL histogram) |
| `04_distillation_training.ipynb` | Train on 1000 examples, watch each loss component converge |
| `05_evaluation_comparison.ipynb` | Final comparison table + bar charts across all 4 conditions |

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
├── configs/distillation_config.yaml   <- all hyperparameters
├── src/
│   ├── data/          <- SQuAD loading, chunking, formatting
│   ├── rag/           <- ChromaDB store, MiniLM embedder, retriever
│   ├── teacher/       <- frozen teacher + RAG-augmented teacher
│   ├── student/       <- trainable student
│   ├── distillation/
│   │   ├── loss.py        <- RADLoss (5 components)
│   │   ├── gating.py      <- retrieval utility, RUG + TRF gates
│   │   ├── curriculum.py  <- retrieval-utility curriculum sampler
│   │   └── trainer.py     <- RADTrainer
│   └── evaluation/
│       ├── evaluator.py   <- EM, F1, BLEU-4, comparison table
│       ├── probe.py       <- Knowledge Retention Probe
│       └── calibration.py <- ECE, sequence confidence
├── scripts/           <- 4-phase pipeline (build -> labels -> train -> eval)
├── notebooks/         <- progressive learning arc
└── tests/             <- 72 unit tests (CI-safe, no GPU required)
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
