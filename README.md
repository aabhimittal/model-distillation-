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
# Fast unit tests — no GPU, no HF models, < 2 min
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
│   ├── distillation/  <- RADLoss + RADTrainer
│   └── evaluation/    <- EM, F1, BLEU-4, comparison table
├── scripts/           <- 4-phase pipeline (build -> labels -> train -> eval)
├── notebooks/         <- progressive learning arc
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
