"""Training and validation engine utilities."""

from typing import Any, Dict, Optional
import os
import time
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses.registry import build_loss, default_loss_for_model
from utils.shared import normalize_model_name


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Adam:
    """Create Adam optimizer for the model."""
    return torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer, config: Dict[str, Any]
) -> torch.optim.lr_scheduler.StepLR:
    """Create StepLR learning rate scheduler."""
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config.get("lr_step_size", 10)),
        gamma=float(config.get("lr_gamma", 0.1)),
    )


def create_criterion(model_name: str, config: Dict[str, Any]) -> Optional[torch.nn.Module]:
    """Create loss function based on model type or explicit config."""
    model_name = normalize_model_name(model_name)
    loss_name = config.get("loss") or default_loss_for_model(model_name)
    return build_loss(loss_name, config)


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    logger: logging.Logger,
    log_interval: int = 100,
    model_name: str = "Model",
) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    seen_batches = 0
    num_batches = len(train_loader)

    progress_bar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}", leave=True)
    start_time = time.time()

    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        loss = model.train_step(batch)
        if loss is None:
            continue

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        seen_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg": f"{(total_loss / seen_batches) if seen_batches else 0.0:.4f}",
        })

        if batch_idx % log_interval == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"[{model_name}] Epoch: {epoch} [{batch_idx}/{num_batches}] "
                f"Loss: {loss.item():.4f} | Time: {elapsed:.1f}s"
            )

    return total_loss / seen_batches if seen_batches > 0 else 0.0


def validate_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    logger: logging.Logger,
    model_name: str = "Model",
) -> Dict[str, float]:
    """Validate the model on a validation set (pairwise metrics)."""
    model.eval()
    total_loss = 0.0
    seen_batches = 0
    num_batches = len(val_loader)

    hit_at_1_correct = 0
    hit_at_1_total = 0

    progress_bar = tqdm(val_loader, desc=f"[{model_name}] Validating", leave=True)

    with torch.no_grad():
        for batch in progress_bar:
            step = model.val_step(batch)
            if step is None:
                continue
            loss, correct, total = step
            if loss is None:
                continue

            total_loss += loss.item()
            seen_batches += 1
            hit_at_1_correct += int(correct)
            hit_at_1_total += int(total)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / seen_batches if seen_batches > 0 else 0.0

    hit_at_1 = (hit_at_1_correct / hit_at_1_total) if hit_at_1_total > 0 else 0.0

    logger.info(f"[{model_name}] Validation Loss: {avg_loss:.4f}")
    logger.info(f"[{model_name}] Validation Hit@1: {hit_at_1:.4f}")

    return {"val_loss": avg_loss, "Hit@1": hit_at_1}


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    config: Dict[str, Any],
    output_dir: str,
    logger: logging.Logger,
    start_epoch: int = 0,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Main training loop for the model."""
    num_epochs = int(config.get("epochs", 30))
    log_interval = int(config.get("log_interval", 100))
    save_interval = int(config.get("save_interval", 10))
    early_stopping_patience = int(config.get("early_stopping_patience", 5))

    best_val_hit1 = 0.0
    patience_counter = 0
    training_history = []

    logger.info(f"[{model_name}] Starting training for {num_epochs} epochs")
    logger.info(f"[{model_name}] Early stopping patience: {early_stopping_patience}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer,
            epoch, logger,
            log_interval=log_interval, model_name=model_name
        )

        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"[{model_name}] Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Time: {epoch_time:.1f}s"
        )

        val_metrics = validate_epoch(
            model, val_loader, logger,
            model_name=model_name
        )
        val_hit1 = val_metrics.get("Hit@1", 0.0)
        val_loss = val_metrics.get("val_loss", float("inf"))

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_hit1": val_hit1,
            "val_loss": val_loss,
            "learning_rate": current_lr,
        }
        training_history.append(epoch_info)

        if val_hit1 > best_val_hit1:
            best_val_hit1 = val_hit1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_hit1": val_hit1,
                    "val_loss": val_loss,
                    "config": config,
                    "model_name": model_name,
                },
                os.path.join(output_dir, "best_model.pt"),
            )
            logger.info(f"[{model_name}] Saved best model with Hit@1 = {val_hit1:.4f}")
        else:
            patience_counter += 1
            logger.info(
                f"[{model_name}] No improvement for {patience_counter}/{early_stopping_patience} epochs"
            )

            if patience_counter >= early_stopping_patience:
                logger.info(
                    f"[{model_name}] Early stopping triggered after {epoch+1} epochs. "
                    f"Best Hit@1: {best_val_hit1:.4f}"
                )
                break

        if (epoch + 1) % save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_hit1": val_hit1,
                    "val_loss": val_loss,
                    "config": config,
                    "model_name": model_name,
                },
                os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt"),
            )
            logger.info(f"[{model_name}] Saved checkpoint at epoch {epoch+1}")

    logger.info(f"[{model_name}] Training completed. Best Val Hit@1: {best_val_hit1:.4f}")

    return {
        "best_val_hit1": best_val_hit1,
        "training_history": training_history,
    }


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> int:
    """Load model and optimizer state from checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return int(checkpoint.get("epoch", -1)) + 1
    return 0
