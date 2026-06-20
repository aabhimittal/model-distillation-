"""RADTrainer — orchestrates the RAG-augmented distillation training loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.student.student_model import StudentModel
from src.teacher.rag_teacher import RAGTeacher
from .loss import RADLoss


def _gpu_memory_check(student: StudentModel, rag_teacher: RAGTeacher) -> None:
    """Warn if VRAM is low; assert student and teacher are on the same device."""
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_gb < 12:
            print(
                f"WARNING: {total_gb:.1f} GB VRAM detected. "
                "Consider reducing batch_size to 4 and gradient_accumulation_steps to 8."
            )
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB VRAM)")

    teacher_device = rag_teacher.teacher.device
    student_device = student.device
    if teacher_device != student_device:
        # This is only a soft warning — device_map="auto" may spread layers across devices
        print(
            f"NOTE: student on '{student_device}', teacher on '{teacher_device}'. "
            "Logits will be moved to student device during loss computation."
        )


class RADTrainer:
    def __init__(
        self,
        student: StudentModel,
        rag_teacher: RAGTeacher,
        loss_fn: RADLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        learning_rate: float = 3e-4,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        logging_steps: int = 50,
        eval_steps: int = 250,
        save_steps: int = 500,
        gradient_accumulation_steps: int = 4,
        disable_cra: bool = False,
    ):
        self.student = student
        self.rag_teacher = rag_teacher
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16 and torch.cuda.is_available()
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.grad_accum = gradient_accumulation_steps
        self.disable_cra = disable_cra

        _gpu_memory_check(student, rag_teacher)

        self.optimizer = AdamW(student.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scaler = GradScaler(enabled=self.fp16)
        self.history: list[dict] = []
        self.global_step = 0

    def _build_decoder_input(self, labels: torch.Tensor) -> torch.Tensor:
        """Shift labels right to build decoder_input_ids (teacher forcing)."""
        pad_id = self.student.tokenizer.pad_token_id
        decoder_input = labels.clone()
        decoder_input[decoder_input == -100] = pad_id
        bos = torch.full((labels.size(0), 1), pad_id, dtype=torch.long)
        decoder_input = torch.cat([bos, decoder_input[:, :-1]], dim=1)
        return decoder_input

    def train_epoch(self, epoch: int) -> float:
        self.student.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            questions = batch["question_text"]

            decoder_input_ids = self._build_decoder_input(labels)

            # Teacher forward (no grad) — RAG, bare, and negative in one call
            rag_logits, bare_logits, neg_logits = self.rag_teacher.get_all_logits(
                questions, decoder_input_ids
            )

            if self.disable_cra:
                neg_logits = bare_logits

            # Student forward (with grad)
            with autocast(enabled=self.fp16):
                student_logits = self.student.forward(
                    input_ids, attention_mask, decoder_input_ids
                )
                losses = self.loss_fn(
                    student_logits=student_logits.float(),
                    rag_teacher_logits=rag_logits.float().to(self.student.device),
                    bare_teacher_logits=bare_logits.float().to(self.student.device),
                    neg_teacher_logits=neg_logits.float().to(self.student.device),
                    labels=labels.to(self.student.device),
                )
                loss = losses["total"] / self.grad_accum

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.logging_steps == 0:
                    log_entry = {
                        "step": self.global_step,
                        "epoch": epoch + 1,
                        "loss": losses["total"].item(),
                        "L_RAG": losses["L_RAG"].item(),
                        "L_KL": losses["L_KL"].item(),
                        "L_CRA": losses["L_CRA"].item(),
                        "L_CE": losses["L_CE"].item(),
                    }
                    self.history.append(log_entry)
                    print(
                        f"  step {self.global_step}: loss={log_entry['loss']:.4f}  "
                        f"L_RAG={log_entry['L_RAG']:.4f}  L_KL={log_entry['L_KL']:.4f}  "
                        f"L_CRA={log_entry['L_CRA']:.4f}  L_CE={log_entry['L_CE']:.4f}"
                    )

                if self.global_step % self.save_steps == 0:
                    self._save(f"checkpoint-{self.global_step}")

            total_loss += losses["total"].item()

        return total_loss / len(self.train_loader)

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}\n")

        self._save("final")
        self._save_history()

    def _save(self, tag: str) -> None:
        save_path = self.output_dir / tag
        self.student.save(str(save_path))
        print(f"Saved checkpoint to {save_path}")

    def _save_history(self) -> None:
        history_path = self.output_dir / "loss_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Loss history saved to {history_path}")
