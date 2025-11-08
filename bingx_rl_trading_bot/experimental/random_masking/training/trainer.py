"""
Trainer for Random Masking Candle Predictor

Complete training pipeline with:
- Multi-task learning
- Learning rate scheduling
- Checkpointing
- Logging (TensorBoard)
- Early stopping
- Gradient clipping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
from tqdm import tqdm
from loguru import logger
from collections import defaultdict

from .losses import combined_loss


class Trainer:
    """
    Trainer for candle prediction model with random masking curriculum
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer

        Args:
            model: CandlePredictor model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Configuration dict with training parameters
            device: Device for training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Loss weights
        self.loss_weights = config.get('loss_weights', {
            'mse': 1.0,
            'directional': 0.3,
            'volatility': 0.2,
            'uncertainty': 0.1
        })

        # Training config
        self.epochs = config.get('epochs', 200)
        self.patience = config.get('patience', 20)
        self.gradient_clip = config.get('gradient_clip', 1.0)
        self.val_freq = config.get('val_freq', 1)

        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = config.get('save_freq', 10)

        # Logging
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_freq = config.get('log_freq', 100)

        # TensorBoard
        if config.get('use_tensorboard', True):
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        logger.info(f"Initialized Trainer:")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - Optimizer: {type(self.optimizer).__name__}")
        logger.info(f"  - Scheduler: {type(self.scheduler).__name__}")
        logger.info(f"  - Epochs: {self.epochs}")
        logger.info(f"  - Patience: {self.patience}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        optimizer_name = self.config.get('optimizer', 'AdamW')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)

        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.config.get('optimizer_params', {}).get('betas', (0.9, 0.999))
            )
        elif optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'CosineAnnealing')

        if scheduler_name == 'CosineAnnealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 200),
                eta_min=self.config.get('scheduler_params', {}).get('eta_min', 1e-6)
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        elif scheduler_name == 'None':
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        return scheduler

    def train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            avg_loss: Average training loss
            metrics: Dict of training metrics
        """
        self.model.train()

        total_loss = 0.0
        metrics = defaultdict(float)
        n_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}')

        for batch_idx, batch in enumerate(pbar):
            # Unpack batch - handle both masking-enabled and baseline formats
            if len(batch) == 5:
                # Masking enabled: full 5-tuple
                masked_seq, target, mask_pos, task_types, future = batch
            elif len(batch) == 2:
                # Masking disabled (baseline): 2-tuple
                sequence, target = batch
                masked_seq = sequence
                # Create dummy mask positions (no masking)
                batch_size, seq_len = sequence.shape[0], sequence.shape[1]
                mask_pos = torch.zeros(batch_size, seq_len, dtype=torch.bool)
                # All forecasting tasks
                task_types = ['forecast'] * batch_size
                # For baseline, future = target
                future = target
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            # Move to device
            masked_seq = masked_seq.to(self.device)
            target = target.to(self.device)
            mask_pos = mask_pos.to(self.device)
            future = future.to(self.device)

            # Forward pass
            predictions, uncertainty = self.model(
                masked_seq,
                mask_positions=mask_pos,
                task_types=task_types,
                return_uncertainty=True
            )

            # Calculate loss
            loss, loss_dict = combined_loss(
                predictions, target, mask_pos, task_types,
                uncertainty, self.loss_weights
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            # Optimizer step
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                metrics[key] += value

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

            # Logging
            if self.writer and (self.global_step % self.log_freq == 0):
                self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)

            self.global_step += 1

        # Average metrics
        avg_loss = total_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in metrics.items()}

        return avg_loss, avg_metrics

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict]:
        """
        Validate on validation set

        Returns:
            avg_loss: Average validation loss
            metrics: Dict of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_targets = []

        for batch in tqdm(self.val_loader, desc='Validation', leave=False):
            # Unpack batch (validation uses forecasting only)
            sequence, future = batch

            # Move to device
            sequence = sequence.to(self.device)
            future = future.to(self.device)

            # Create forecasting mask (mask last pred_len positions)
            batch_size, seq_len, n_features = sequence.shape
            pred_len = future.shape[1]

            # Mask future positions
            mask_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
            mask_positions[:, -pred_len:] = True

            # Task types (all forecasting)
            task_types = ['forecast'] * batch_size

            # Forward pass
            predictions, uncertainty = self.model(
                sequence,
                mask_positions=mask_positions,
                task_types=task_types,
                return_uncertainty=True
            )

            # Extract predictions for future section
            pred_future = predictions[:, -pred_len:, :]

            # Loss
            loss = nn.functional.mse_loss(pred_future, future)
            total_loss += loss.item()

            # Collect for metrics
            all_predictions.append(pred_future.cpu())
            all_targets.append(future.cpu())

        # Calculate metrics
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        metrics = {
            'mse': nn.functional.mse_loss(predictions, targets).item(),
            'mae': nn.functional.l1_loss(predictions, targets).item(),
            'directional_accuracy': self._calculate_directional_accuracy(predictions, targets)
        }

        avg_loss = total_loss / len(self.val_loader)

        return avg_loss, metrics

    def _calculate_directional_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Calculate directional accuracy for close price

        Args:
            predictions: Predicted values (batch, seq_len, n_features)
            targets: Ground truth (batch, seq_len, n_features)

        Returns:
            Directional accuracy (fraction correct)
        """
        close_idx = 3  # Assuming OHLCV order

        # Get close prices
        pred_close = predictions[:, :, close_idx]
        target_close = targets[:, :, close_idx]

        # Calculate price changes
        pred_change = pred_close[:, 1:] - pred_close[:, :-1]
        target_change = target_close[:, 1:] - target_close[:, :-1]

        # Direction
        pred_direction = torch.sign(pred_change)
        target_direction = torch.sign(target_change)

        # Accuracy
        correct = (pred_direction == target_direction).float()
        accuracy = correct.mean().item()

        return accuracy

    def train(self):
        """
        Main training loop

        Returns:
            History dict with training metrics
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }

        logger.info(f"Starting training for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch

            # Train
            train_loss, train_metrics = self.train_epoch(epoch)

            # Validate
            if epoch % self.val_freq == 0:
                val_loss, val_metrics = self.validate()

                # Log
                logger.info(f"Epoch {epoch}/{self.epochs}:")
                logger.info(f"  Train Loss: {train_loss:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}")
                logger.info(f"  Val MSE: {val_metrics['mse']:.4f}")
                logger.info(f"  Val Dir Acc: {val_metrics['directional_accuracy']:.4f}")

                # TensorBoard
                if self.writer:
                    self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
                    self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f'epoch/val_{key}', value, epoch)

                    # Learning rate
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('epoch/learning_rate', current_lr, epoch)

                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_metrics'].append(val_metrics)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    self.epochs_without_improvement = 0
                    logger.info(f"  ✅ New best model! Val Loss: {val_loss:.4f}")
                else:
                    self.epochs_without_improvement += 1

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Periodic checkpoint
            if epoch % self.save_freq == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience: {self.patience})")
                break

        # Final cleanup
        if self.writer:
            self.writer.close()

        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        return history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")


if __name__ == '__main__':
    # Test trainer (minimal example)
    print("=" * 60)
    print("Testing Trainer")
    print("=" * 60)

    from torch.utils.data import TensorDataset

    # Create dummy data
    n_samples = 1000
    seq_len = 100
    input_dim = 15

    # Dummy training data
    X_train = torch.randn(n_samples, seq_len, input_dim)
    y_train = torch.randn(n_samples, 10, input_dim)  # 10 future steps

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Dummy validation data
    X_val = torch.randn(200, seq_len, input_dim)
    y_val = torch.randn(200, 10, input_dim)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Create model
    from models.predictor import CandlePredictor

    model = CandlePredictor(
        input_dim=input_dim,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        ff_dim=512,
        dropout=0.1
    )

    # Training config
    config = {
        'epochs': 3,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'patience': 10,
        'gradient_clip': 1.0,
        'val_freq': 1,
        'save_freq': 1,
        'log_freq': 10,
        'use_tensorboard': False,
        'checkpoint_dir': 'test_checkpoints',
        'log_dir': 'test_logs'
    }

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device='cpu'
    )

    print("\nTrainer initialized successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    print("\n" + "=" * 60)
    print("Trainer test passed! ✅")
    print("=" * 60)
