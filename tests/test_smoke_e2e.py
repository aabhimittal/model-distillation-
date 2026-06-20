"""
CPU end-to-end smoke test for the full RAD pipeline.

Uses flan-t5-small as *both* teacher and student (tiny, same vocab),
a 4-example toy dataset, and an ephemeral ChromaDB — no GPU required.

Runtime: ~5-10 min on CPU (model download on first run, then cached).

Run with:
    pytest tests/test_smoke_e2e.py -v --timeout=600 -m slow
"""

import pytest
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Toy data — mimics SQuAD structure without downloading anything
# ---------------------------------------------------------------------------
TOY_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "context": "Paris is the capital and most populous city of France.",
        "answer": "Paris",
        "id": "q001",
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "context": "Romeo and Juliet is a play written by William Shakespeare.",
        "answer": "William Shakespeare",
        "id": "q002",
    },
    {
        "question": "What is the speed of light?",
        "context": "The speed of light in vacuum is approximately 299,792 kilometres per second.",
        "answer": "299,792 kilometres per second",
        "id": "q003",
    },
    {
        "question": "Where is the Eiffel Tower located?",
        "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
        "answer": "Paris",
        "id": "q004",
    },
]

MODEL_NAME = "google/flan-t5-small"  # ~300 MB; also used as teacher for speed
MAX_INPUT = 128
MAX_TARGET = 16


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def embedder():
    from src.rag.embedder import Embedder
    return Embedder("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def chroma_store_with_passages(embedder):
    from src.rag.chroma_store import ChromaStore
    passages = [ex["context"] for ex in TOY_EXAMPLES]
    store = ChromaStore()  # ephemeral
    embs = embedder.embed_passages(passages, batch_size=4)
    metadata = [{"idx": i} for i in range(len(passages))]
    store.add_documents(passages, embs, metadata)
    return store


@pytest.fixture(scope="module")
def retriever(embedder, chroma_store_with_passages):
    from src.rag.retriever import Retriever
    return Retriever(embedder, chroma_store_with_passages)


@pytest.fixture(scope="module")
def teacher_model():
    from src.teacher.teacher_model import TeacherModel
    return TeacherModel(MODEL_NAME, device="cpu")


@pytest.fixture(scope="module")
def rag_teacher(teacher_model, retriever):
    from src.teacher.rag_teacher import RAGTeacher
    return RAGTeacher(teacher_model, retriever, max_input_length=MAX_INPUT)


@pytest.fixture(scope="module")
def student_model():
    from src.student.student_model import StudentModel
    return StudentModel(MODEL_NAME, device="cpu")


# ---------------------------------------------------------------------------
# Helper: build a small DataLoader from toy examples
# ---------------------------------------------------------------------------
def _make_batch(tokenizer):
    input_ids_list, attn_list, label_list, questions, answer_texts, ids = [], [], [], [], [], []
    for ex in TOY_EXAMPLES:
        enc = tokenizer(
            f"question: {ex['question']}",
            max_length=MAX_INPUT,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tgt = tokenizer(
            ex["answer"],
            max_length=MAX_TARGET,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = tgt["input_ids"].squeeze()
        labels[labels == tokenizer.pad_token_id] = -100
        input_ids_list.append(enc["input_ids"].squeeze())
        attn_list.append(enc["attention_mask"].squeeze())
        label_list.append(labels)
        questions.append(ex["question"])
        answer_texts.append(ex["answer"])
        ids.append(ex["id"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_list),
        "labels": torch.stack(label_list),
        "question_text": questions,
        "answer_text": answer_texts,
        "example_id": ids,
    }


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_rag_pipeline_produces_correct_shapes(embedder, chroma_store_with_passages, retriever):
    """Embedder → ChromaStore → Retriever all work together."""
    assert chroma_store_with_passages.collection_size() == 4

    results = retriever.retrieve("capital of France", top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, str) and len(r) > 0 for r in results)


@pytest.mark.slow
def test_rag_teacher_logit_shapes(rag_teacher, tokenizer):
    """RAGTeacher produces three logit tensors of shape (B, L_dec, V)."""
    batch = _make_batch(tokenizer)
    questions = batch["question_text"]
    labels = batch["labels"]

    pad_id = tokenizer.pad_token_id
    dec = labels.clone()
    dec[dec == -100] = pad_id
    bos = torch.full((labels.size(0), 1), pad_id, dtype=torch.long)
    decoder_input_ids = torch.cat([bos, dec[:, :-1]], dim=1)

    rag_logits, bare_logits, neg_logits = rag_teacher.get_all_logits(questions, decoder_input_ids)

    B = len(questions)
    L = decoder_input_ids.size(1)

    assert rag_logits.shape[0] == B
    assert bare_logits.shape[0] == B
    assert neg_logits.shape[0] == B
    assert rag_logits.shape[1] == L
    assert rag_logits.shape == bare_logits.shape == neg_logits.shape


@pytest.mark.slow
def test_teacher_params_have_no_grad(rag_teacher, tokenizer):
    """Teacher weights must remain frozen throughout (no grad)."""
    for param in rag_teacher.teacher.model.parameters():
        assert param.grad is None
        assert not param.requires_grad


@pytest.mark.slow
def test_rad_loss_with_real_logits(rag_teacher, student_model, tokenizer):
    """RADLoss produces valid components when fed real teacher and student logits."""
    from src.distillation.loss import RADLoss

    batch = _make_batch(tokenizer)
    questions = batch["question_text"]
    labels = batch["labels"]

    pad_id = tokenizer.pad_token_id
    dec = labels.clone()
    dec[dec == -100] = pad_id
    bos = torch.full((labels.size(0), 1), pad_id, dtype=torch.long)
    decoder_input_ids = torch.cat([bos, dec[:, :-1]], dim=1)

    rag_logits, bare_logits, neg_logits = rag_teacher.get_all_logits(questions, decoder_input_ids)

    student_logits = student_model.forward(
        batch["input_ids"], batch["attention_mask"], decoder_input_ids
    )

    loss_fn = RADLoss(temperature=2.0)
    result = loss_fn(
        student_logits=student_logits.float(),
        rag_teacher_logits=rag_logits.float(),
        bare_teacher_logits=bare_logits.float(),
        neg_teacher_logits=neg_logits.float(),
        labels=labels,
    )

    assert result["total"].item() > 0
    for key in ("L_RAG", "L_KL", "L_CRA", "L_CE"):
        assert result[key].item() >= 0, f"{key} is negative"


@pytest.mark.slow
def test_one_training_step_updates_student(rag_teacher, student_model, tokenizer):
    """A single RADTrainer step completes and records a loss entry."""
    import copy
    from torch.utils.data import TensorDataset
    from src.distillation.loss import RADLoss
    from src.distillation.trainer import RADTrainer

    batch = _make_batch(tokenizer)

    # Snapshot student weights before training
    param_before = copy.deepcopy(
        next(iter(student_model.model.parameters())).detach().clone()
    )

    loss_fn = RADLoss(temperature=2.0)

    class SingleBatchLoader:
        """Yields the same batch once per epoch."""
        def __iter__(self):
            yield batch
        def __len__(self):
            return 1

    trainer = RADTrainer(
        student=student_model,
        rag_teacher=rag_teacher,
        loss_fn=loss_fn,
        train_loader=SingleBatchLoader(),
        val_loader=SingleBatchLoader(),
        output_dir="/tmp/rad_smoke_test",
        learning_rate=1e-4,
        fp16=False,   # CPU
        logging_steps=1,
        eval_steps=999,
        save_steps=999,
        gradient_accumulation_steps=1,
    )

    avg_loss = trainer.train_epoch(epoch=0)

    assert avg_loss > 0
    assert len(trainer.history) >= 1

    # Student weights should have changed
    param_after = next(iter(student_model.model.parameters())).detach().clone()
    assert not torch.allclose(param_before, param_after), "Student weights did not update"


@pytest.mark.slow
def test_evaluator_runs_on_generated_outputs(student_model, tokenizer):
    """Evaluator.evaluate() works on outputs generated by the student model."""
    from src.evaluation.evaluator import Evaluator

    questions = [ex["question"] for ex in TOY_EXAMPLES]
    references = [ex["answer"] for ex in TOY_EXAMPLES]

    enc = tokenizer(
        [f"question: {q}" for q in questions],
        max_length=MAX_INPUT,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    predictions = student_model.generate_text(
        enc["input_ids"], enc["attention_mask"], max_new_tokens=MAX_TARGET
    )

    evaluator = Evaluator()
    metrics = evaluator.evaluate(predictions, references)

    assert "exact_match" in metrics
    assert "f1" in metrics
    assert "bleu4" in metrics
    assert 0.0 <= metrics["exact_match"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["n"] == len(TOY_EXAMPLES)
