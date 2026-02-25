"""Training entrypoint for MATCH-A image matching models."""

import os
import argparse

from utils.logger import setup_logger
from utils.shared import select_transforms, normalize_model_name
from utils.pipeline import (
    set_seed,
    resolve_device,
    resolve_model_name,
    load_config_with_overrides,
    build_model,
    build_dataset,
    build_dataloader,
)
from utils.engine import (
    create_optimizer,
    create_scheduler,
    create_criterion,
    train_model,
    load_checkpoint,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train MATCH-A models")

    parser.add_argument("--config", type=str, default="configs/contrastive_clip.yaml",
                        help="Path to config file")
    parser.add_argument("--csv", type=str, default="local_matcha_dataset/data_splits.csv",
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
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints and logs")

    return parser.parse_args()


def main():
    """Main function to start training."""
    args = parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    config = load_config_with_overrides(args.config, args)
    print(f"Loaded configuration from {args.config}")
    print(
        "Config values: "
        f"batch_size={config.get('batch_size')}, "
        f"lr={config.get('lr')}, "
        f"epochs={config.get('epochs')}"
    )

    model_name = resolve_model_name(args.model, config)
    model_name = normalize_model_name(model_name)
    print(f"Using model: {model_name}")

    output_dir = args.output_dir
    if not output_dir:
        config_stem = os.path.splitext(os.path.basename(args.config))[0]
        output_dir = os.path.join("outputs", config_stem)

    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(log_file=os.path.join(output_dir, "train.log"))
    logger.info(f"Starting training for model: {model_name}")
    logger.info(f"Configuration: {config}")

    transform = select_transforms(model_name, config)
    model = build_model(model_name, config, device)

    train_dataset = build_dataset(model_name, args.csv, "train", transform=transform, config=config)
    val_dataset = build_dataset(model_name, args.csv, "val", transform=transform, config=config)

    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 4))
    train_loader = build_dataloader(
        train_dataset, model_name, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = build_dataloader(
        val_dataset, model_name, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = model.to(device)
    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"Target device: {device}")

    criterion = create_criterion(model_name, config)
    model.set_loss(criterion)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer, device)
        logger.info(f"Resumed from epoch {start_epoch}")

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_dir=output_dir,
        logger=logger,
        start_epoch=start_epoch,
        model_name=model_name,
    )

    logger.info(f"Training completed. Best Val Hit@1: {results['best_val_hit1']:.4f}")
    return results


if __name__ == "__main__":
    main()
