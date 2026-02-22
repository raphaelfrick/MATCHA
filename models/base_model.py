"""Base model class for image matching models.

This module provides the base class for all model architectures used in the
MATCH-A framework, including common training and validation logic.
"""

import os
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
from PIL import Image


class BaseModel(nn.Module):
    """Base class for all image matching models.
    
    This class provides common functionality for training, validation, and
    prediction that is shared across different model architectures.
    
    Args:
        config: Configuration dictionary containing model hyperparameters.
        device: torch.device to run the model on.
    
    Attributes:
        config: Configuration dictionary.
        device: Device the model runs on.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.to(device)

    def _cast_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Cast configuration values to appropriate types with defaults.
        
        This method ensures all configuration values have the correct types
        and provides sensible defaults for missing values.
        
        Args:
            config: Raw configuration dictionary.
            
        Returns:
            Configuration dictionary with typed values and defaults.
        """
        return {
            'embedding_dim': int(config.get('embedding_dim', 128)),
            'margin': float(config.get('margin', 0.2)),
            'lr': float(config.get('lr', 1e-4)),
            'threshold': float(config.get('threshold', 0.7)),
            'batch_size': int(config.get('batch_size', 32)),
            'epochs': int(config.get('epochs', 20)),
            'early_stopping_patience': int(config.get('early_stopping_patience', 5)),
            **config
        }

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")

    def train_step(self, batch):
        raise NotImplementedError("Subclasses must implement train_step()")

    def val_step(self, batch):
        raise NotImplementedError("Subclasses must implement val_step()")

    def train_model(self, train_loader, val_loader, save_path):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        best_f1 = 0
        patience = 0

        for epoch in range(self.config['epochs']):
            self.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                loss = self.train_step(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1} - Train Loss: {total_loss / len(train_loader):.4f}")
            val_f1 = self.validate(val_loader)
            print(f"Epoch {epoch+1} - Val F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience = 0
                torch.save(self.state_dict(), os.path.join(save_path, f"best_{self.__class__.__name__}.pt"))
                print("✅ Saved new best model.")
            else:
                patience += 1
                if patience >= self.config['early_stopping_patience']:
                    print("⏹️ Early stopping triggered.")
                    break

    def validate(self, val_loader):
        self.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels, preds = self.val_step(batch)
                y_true.extend(labels)
                y_pred.extend(preds)
        return f1_score(y_true, y_pred)

    def predict(self, query_paths, auth_loader):
        self.eval()
        results = {}
        with torch.no_grad():
            for q_path in tqdm(query_paths, desc="Predicting"):
                q_img = Image.open(q_path).convert("RGB")
                q_tensor = auth_loader.dataset.transform(q_img).unsqueeze(0).to(self.device)
                q_emb = self(q_tensor)

                best_score, best_match = -1, None
                for db_imgs, db_paths in auth_loader:
                    db_imgs = db_imgs.to(self.device)
                    db_embs = self(db_imgs)
                    sims = self.match_score(q_emb.repeat(len(db_imgs), 1), db_embs)
                    max_idx = torch.argmax(sims).item()
                    if sims[max_idx] > best_score:
                        best_score = sims[max_idx].item()
                        best_match = db_paths[max_idx]
                results[q_path] = {
                    "match": best_match,
                    "score": best_score,
                    "is_match": best_score > self.config['threshold']
                }
        return results
