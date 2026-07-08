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
L_RAD = α·L_RAG + β·L_KL + γ·L_CRA + δ·L_CE

L_RAG = KL( p_S(y|q) || p_T(y|q, c_RAG) )        # student vs RAG-teacher
L_KL  = KL( p_S(y|q) || p_T(y|q) )                # student vs bare teacher
L_CRA = max(0, margin - KL(p_T(y|q,c+) || p_T(y|q,c-))) # contrastive retrieval alignment
L_CE  = CrossEntropy( p_S(y|q), y* )              # hard-label grounding

Defaults: α=0.5, β=0.2, γ=0.1, δ=0.2, T=4.0
```

**L_CRA is the key novelty.** It detects and penalises *degenerate retrieval* — the failure mode where the teacher ignores its retrieved context (because it already knows the answer parametrically). L_CRA enforces that the RAG-teacher's distribution is margin-separated from its negative-context distribution, guaranteeing that L_RAG carries a genuine signal. All KL terms scale by T^2 to preserve gradient magnitude (Hinton et al., 2015).

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

### Ablation: disable L_CRA to see its contribution

```bash
python scripts/train_student.py --disable-cra
```

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
# Fast unit tests — no GPU, no HF models, < 2 min
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
│   ├── distillation/  <- RADLoss + RADTrainer  (RAD)
│   └── evaluation/    <- EM, F1, BLEU-4, comparison table  (RAD)
├── scripts/
│   ├── finetune_distill.py        <- Track A entrypoint
│   ├── infer.py                   <- run a fine-tuned adapter
│   └── build_vector_db.py …       <- Track B 4-phase pipeline
├── notebooks/         <- colab_finetune_distill.ipynb (Track A)
└── tests/             <- unit tests (CI-safe, no GPU required)
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
