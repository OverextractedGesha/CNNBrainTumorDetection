"""Training loop, two-phase backbone freezing, warmup/cosine LR scheduling, and checkpoints."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class TrainHistory:
    """Recorded epoch-by-epoch metrics and best checkpoint metadata from training."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    best_checkpoint_path: str | None = None


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts (images, targets) from tuple/list or dictionary DataLoader batches."""
    if isinstance(batch, dict):
        return batch["image"], batch["label"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError(f"Unsupported batch structure of type {type(batch).__name__}.")


def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Builds a warmup + cosine decay LambdaLR schedule safe for any ``epochs >= 1``."""
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}.")
    total_epochs = max(1, int(epochs))
    warmup = max(0, int(warmup_epochs))
    cosine_span = max(1, total_epochs - warmup)

    def _lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = min(1.0, float(epoch - warmup) / float(cosine_span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


class Trainer:
    """Two-phase trainer with backbone freeze/unfreeze, AMP on CUDA, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device | None = None,
        epochs: int = 15,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 2,
        freeze_backbone_epochs: int = 2,
        early_stopping_patience: int = 5,
        checkpoint_dir: str | Path = "checkpoints",
        use_amp: bool = True,
        grad_clip_norm: float | None = 1.0,
        config: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_epochs = int(warmup_epochs)
        self.freeze_backbone_epochs = int(freeze_backbone_epochs)
        self.early_stopping_patience = int(early_stopping_patience)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.grad_clip_norm = grad_clip_norm
        self.config = config
        self.verbose = bool(verbose)

        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}.")
        if not (math.isfinite(self.lr) and self.lr > 0.0):
            raise ValueError(f"lr must be positive and finite, got {self.lr}.")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {self.warmup_epochs}.")
        if self.freeze_backbone_epochs < 0:
            raise ValueError(
                f"freeze_backbone_epochs must be non-negative, got {self.freeze_backbone_epochs}."
            )
        if self.early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience must be non-negative, got {self.early_stopping_patience}."
            )
        if self.grad_clip_norm is not None and not (
            math.isfinite(self.grad_clip_norm) and self.grad_clip_norm >= 0.0
        ):
            raise ValueError(
                f"grad_clip_norm must be non-negative and finite, got {self.grad_clip_norm}."
            )

        if optimizer is None:
            self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optimizer

        self.scheduler = _build_warmup_cosine_scheduler(
            optimizer=self.optimizer,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        # Enable AMP autocast and GradScaler strictly on CUDA devices
        self.amp_enabled = bool(use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.amp_enabled else None

    def _autocast_context(self) -> Any:
        if self.amp_enabled:
            return torch.amp.autocast("cuda")
        return nullcontext()

    def _apply_freeze_schedule(self, epoch: int) -> None:
        if epoch < self.freeze_backbone_epochs:
            if hasattr(self.model, "freeze_backbone"):
                self.model.freeze_backbone()
        elif epoch == self.freeze_backbone_epochs:
            if hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone()

    def _train_one_epoch(
        self, train_loader: DataLoader, epoch: int = 0
    ) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        batch_iter: Any = (
            tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                leave=False,
                unit="batch",
            )
            if self.verbose
            else train_loader
        )

        for batch in batch_iter:
            images, targets = _unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).long()
            batch_size = int(targets.shape[0])

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_context():
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                self.optimizer.step()

            total_loss += float(loss.item()) * batch_size
            preds = torch.argmax(logits.detach(), dim=-1)
            total_correct += int((preds == targets).sum().item())
            total_samples += batch_size

        if total_samples == 0:
            return 0.0, 0.0
        return total_loss / total_samples, float(total_correct) / float(total_samples)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Evaluates model on ``loader`` and returns ``(avg_loss, accuracy)``."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in loader:
            images, targets = _unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).long()
            batch_size = int(targets.shape[0])

            with self._autocast_context():
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            total_loss += float(loss.item()) * batch_size
            preds = torch.argmax(logits, dim=-1)
            total_correct += int((preds == targets).sum().item())
            total_samples += batch_size

        if total_samples == 0:
            return 0.0, 0.0
        return total_loss / total_samples, float(total_correct) / float(total_samples)

    @torch.no_grad()
    def predict_logits(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Runs inference over ``loader`` and returns ``(logits, targets)`` NumPy arrays."""
        self.model.eval()
        all_logits: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        for batch in loader:
            images, targets = _unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)

            with self._autocast_context():
                logits = self.model(images)

            all_logits.append(logits.detach().float().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy().astype(np.int64))

        if not all_logits:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
        return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainHistory:
        """Trains the model with two-phase backbone freezing, early stopping, and best-weight restoration."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = self.checkpoint_dir / "best_model.pt"

        history = TrainHistory()
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self._apply_freeze_schedule(epoch)
            phase = "frozen" if epoch < self.freeze_backbone_epochs else "unfrozen"
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            train_loss, train_acc = self._train_one_epoch(train_loader, epoch=epoch)
            val_loss, val_acc = self.evaluate(val_loader)
            self.scheduler.step()

            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.train_acc.append(train_acc)
            history.val_acc.append(val_acc)
            history.lr.append(current_lr)

            is_best = math.isfinite(val_loss) and val_loss < history.best_val_loss
            best_tag = " [best]" if is_best else ""

            if is_best:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                history.best_checkpoint_path = str(best_ckpt_path)
                epochs_without_improvement = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "config": self.config,
                    },
                    best_ckpt_path,
                )
            else:
                epochs_without_improvement += 1

            if self.verbose:
                print(
                    f"Epoch {epoch + 1:02d}/{self.epochs:02d} [{phase}] | "
                    f"lr={current_lr:.2e} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}{best_tag}",
                    flush=True,
                )

            if (
                not is_best
                and self.early_stopping_patience > 0
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                if self.verbose:
                    print(
                        f"Early stopping triggered at epoch {epoch + 1} "
                        f"(no val_loss improvement for {epochs_without_improvement} epochs).",
                        flush=True,
                    )
                break

        if history.best_checkpoint_path is not None:
            load_checkpoint(history.best_checkpoint_path, self.model, device=self.device)

        return history


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Loads saved model weights and metadata from ``checkpoint_path`` into ``model``."""
    target_device = torch.device(device)
    ckpt = torch.load(
        Path(checkpoint_path),
        map_location=target_device,
        weights_only=False,
    )
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        metadata = ckpt
    else:
        model.load_state_dict(ckpt)
        metadata = {"model_state_dict": ckpt}

    model.to(target_device)
    return metadata
