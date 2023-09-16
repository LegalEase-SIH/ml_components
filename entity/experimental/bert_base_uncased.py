import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import AutoTokenizer, BertModel

from entity import config as cfg
from entity.dataset import EntityRecognitionDataset


# Output Data Class
@dataclass
class BertEntityModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]


# Bert Entity Model
class BertEntityModel(nn.Module):
    def __init__(self, hidden_size: int, num_tags: int) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(p=cfg.DROPOUT)
        self.ffwd = nn.Sequential(
            nn.Linear(
                in_features=self.bert.config.hidden_size, out_features=hidden_size
            ),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=num_tags),
        )
        self.num_tags = num_tags
        self.loss_fct = nn.CrossEntropyLoss()

    @staticmethod
    def _compute_loss(
        logits: torch.Tensor, labels: torch.Tensor, num_tags: int
    ) -> torch.Tensor:
        # TODO: Add Loss Function...
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = False,
    ) -> Any[Dict[str, torch.Tensor], BertEntityModelOutput]:
        outputs = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.pooler_output  # [BATCH_SIZE, 768]
        logits = self.ffwd.forward(logits)

        loss = None
        if logits is not None:
            loss = self._compute_loss(
                logits=logits, labels=labels, num_tags=self.num_tags
            )

        if return_dict:
            return {"logits": logits, "loss": loss}

        return BertEntityModelOutput(logits=logits, loss=loss)


# Train One Epoch
def train_one_epoch():
    pass


# Valid One Epoch
@torch.no_grad()
def valid_one_epoch():
    pass


# Main Logic for training
def run_training():
    pass
