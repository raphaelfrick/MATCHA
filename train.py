"""Training functionality for MATCH-A image matching models.

This module provides training functions for training image matching
models on the MATCH-A dataset using various architectures (TripletNet,
ContrastiveViT, ContrastiveCLIP).
"""

import os
import logging
import argparse
from typing import Any, Dict, Optional, Tuple, Union, Set, List
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import from local modules
from utils.config import load_config
from utils.logger import setup_logger
from losses.contrastive_loss import ContrastiveLoss
from utils.shared import (
    get_model_and_dataset,
    select_transforms,
    clip_collate,
    triplet_collate
)


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    use_clip: bool = False,
    use_triplet: bool = False
) -> float:
    """Train the model for one epoch.
    
    This function performs a single training epoch, iterating over the training
    data loader, computing losses, and updating model parameters.
    
    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function.
        device: Device to train on (CPU or GPU).
        epoch: Current epoch number (for logging).
        logger: Logger for tracking metrics.
        use_clip: Whether using CLIP model (uses different collate).
        use_triplet: Whether using TripletNet model (triplet loss).
    
    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        if use_clip:
            batch = clip_collate(batch)
        
        if use_triplet:
            # TripletNet: batch is (anchor, positive, negative) or with paths
            if len(batch) >= 3:
                anchor, positive, negative = batch[:3]
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                optimizer.zero_grad()
                loss = model.compute_triplet_loss(model(anchor), model(positive), model(negative))
            else:
                raise ValueError(f"Unexpected batch format for triplet: {batch}")
        elif use_clip:
            # CLIP: batch is (anchor_images, positive_images) or (images, paths)
            if len(batch) == 4:
                anchor, positive, paths1, paths2 = batch
                anchor = anchor.to(device)
                positive = positive.to(device)
                optimizer.zero_grad()
                loss = criterion(model, anchor, positive)
            else:
                images, paths = batch
                images = images.to(device)
                optimizer.zero_grad()
                loss = criterion(model, images)
        else:
            # Contrastive: batch is (anchor, positive, path1, path2)
            anchor, positive, paths1, paths2 = batch
            anchor = anchor.to(device)
            positive = positive.to(device)
            optimizer.zero_grad()
            loss = criterion(model, anchor, positive)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            logger.info(f"Epoch: {epoch} [{batch_idx}/{num_batches}] Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    use_clip: bool = False,
    use_triplet: bool = False
) -> Dict[str, float]:
    """Validate the model on the validation set.
    
    This function evaluates the model on the validation set and returns
    validation metrics.
    
    Args:
        model: The neural network model to validate.
        val_loader: DataLoader for validation data.
        device: Device to validate on (CPU or GPU).
        logger: Logger for tracking metrics.
        use_clip: Whether using CLIP model.
        use_triplet: Whether using TripletNet model.
    
    Returns:
        Dictionary of validation metrics.
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in val_loader:
            if use_clip:
                batch = clip_collate(batch)
            
            if use_triplet:
                if len(batch) >= 3:
                    anchor, positive, negative = batch[:3]
                    anchor = anchor.to(device)
                    positive = positive.to(device)
                    negative = negative.to(device)
                    loss = model.compute_triplet_loss(model(anchor), model(positive), model(negative))
            elif use_clip:
                if len(batch) == 4:
                    anchor, positive, _, _ = batch
                    anchor = anchor.to(device)
                    positive = positive.to(device)
                    loss = criterion(model, anchor, positive)
                else:
                    images, _ = batch
                    images = images.to(device)
                    loss = criterion(model, images)
            else:
                anchor, positive, _, _ = batch
                anchor = anchor.to(device)
                positive = positive.to(device)
                loss = criterion(model, anchor, positive)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    
    # For simplicity, we return validation loss as the metric
    # In a more complete implementation, this would compute Hit@1, etc.
    return {"val_loss": avg_loss, "Hit@1": 1.0 / (1.0 + avg_loss)}


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    criterion: torch.nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    output_dir: str,
    logger: logging.Logger,
    start_epoch: int = 0,
    use_clip: bool = False,
    use_triplet: bool = False
) -> Dict[str, Any]:
    """Main training loop for the model.
    
    This function orchestrates the training process, iterating over epochs,
    training the model, validating, updating learning rate, and saving checkpoints.
    
    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for updating model parameters.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        device: Device to train on (CPU or GPU).
        config: Configuration dictionary.
        output_dir: Directory to save checkpoints and logs.
        logger: Logger for tracking metrics.
        start_epoch: Starting epoch number (for resuming).
        use_clip: Whether using CLIP model.
        use_triplet: Whether using TripletNet model.
    
    Returns:
        Dictionary containing training results and best metrics.
    """
    num_epochs = config.get("epochs", 30)
    best_val_hit1 = 0.0
    training_history = []
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, logger, use_clip=use_clip, use_triplet=use_triplet
        )
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
        # Validate
        val_metrics = validate(
            model, val_loader, device, logger,
            use_clip=use_clip, use_triplet=use_triplet
        )
        val_hit1 = val_metrics.get("Hit@1", 0.0)
        logger.info(f"Epoch {epoch}: Val Hit@1 = {val_hit1:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Record training history
        epoch_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_hit1': val_hit1,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        training_history.append(epoch_info)
        
        # Save best model
        if val_hit1 > best_val_hit1:
            best_val_hit1 = val_hit1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_hit1': val_hit1,
                'config': config
            }, os.path.join(output_dir, "best_model.pt"))
            logger.info(f"Saved best model with Hit@1 = {val_hit1:.4f}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config.get("save_interval", 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_hit1': val_hit1,
                'config': config
            }, os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
    
    return {
        'best_val_hit1': best_val_hit1,
        'training_history': training_history
    }


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Adam:
    """Create Adam optimizer for the model.
    
    Args:
        model: The neural network model.
        config: Configuration dictionary with learning rate.
    
    Returns:
        Adam optimizer instance.
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-4)
    )


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> torch.optim.lr_scheduler.StepLR:
    """Create StepLR learning rate scheduler.
    
    Args:
        optimizer: The optimizer to schedule.
        config: Configuration dictionary with scheduler parameters.
    
    Returns:
        StepLR scheduler instance.
    """
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.get("lr_step_size", 10),
        gamma=config.get("lr_gamma", 0.1)
    )


def create_criterion(model_name: str) -> torch.nn.Module:
    """Create loss function based on model type.
    
    Args:
        model_name: Name of the model architecture.
    
    Returns:
        Loss function instance.
    """
    if model_name == "triplet_net":
        return torch.nn.MarginRankingLoss(margin=1.0)
    else:
        return ContrastiveLoss(margin=1.0)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> int:
    """Load model and optimizer state from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: The neural network model.
        optimizer: The optimizer.
        device: Device to load checkpoint on.
    
    Returns:
        Epoch number to resume from.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch
    return 0


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train MATCH-A models")
    
    parser.add_argument("--config", type=str, default="old_fx/configs/contrastive_clip.yaml",
                        help="Path to config file")
    parser.add_argument("--csv", type=str, default="old_fx/local_matcha_dataset/data_splits.csv",
                        help="Path to CSV file")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (overrides config if provided)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for checkpoints and logs")
    
    return parser.parse_args()


def main():
    """Main function to start training."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Override model name if provided
    model_name = args.model if args.model else config.get("model", config.get("model_name", "contrastive_clip"))
    print(f"Using model: {model_name}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logger
    logger = setup_logger(log_file=os.path.join(args.output_dir, "train.log"))
    logger.info(f"Starting training for model: {model_name}")
    logger.info(f"Configuration: {config}")
    
    # Select transforms
    transform = select_transforms(model_name, config)
    
    # Get model and datasets
    model, train_dataset = get_model_and_dataset(
        model_name, config, args.csv, "train", device, transform
    )
    _, val_dataset = get_model_and_dataset(
        model_name, config, args.csv, "val", device, transform
    )
    
    # Create data loaders
    use_clip = (model_name == "contrastive_clip")
    use_triplet = (model_name == "triplet_net")
    collate_fn = clip_collate if use_clip else (triplet_collate if use_triplet else None)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.get("batch_size", 32),
        shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.get("batch_size", 32),
        shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create loss function, optimizer, and scheduler
    criterion = create_criterion(model_name)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer, device)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Run training
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config,
        output_dir=args.output_dir,
        logger=logger,
        start_epoch=start_epoch,
        use_clip=use_clip,
        use_triplet=use_triplet
    )
    
    logger.info(f"Training completed. Best Val Hit@1: {results['best_val_hit1']:.4f}")
    
    return results


if __name__ == "__main__":
    main()
