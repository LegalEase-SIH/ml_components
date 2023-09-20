import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel
import pandas as pd

import config as cfg
from dataset import EntityRecognitionDataset


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
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        num_tags: int,
        loss_fct: nn.CrossEntropyLoss,
    ) -> torch.Tensor:
        loss = None
        logits = logits.view(-1, num_tags)  # [BATCH_SIZE * SEQ_LEN, NUM_TAGS]
        labels = labels.view(-1)  # [BATCH_SIZE * SEQ_LEN]
        attention_mask = attention_mask.view(-1)  # [BATCH_SIZE * SEQ_LEN]
        # mask the padding tokens with -100 to ignore when computing it.
        labels = torch.where(
            attention_mask == 1,
            labels,
            torch.tensor(loss_fct.ignore_index, dtype=labels.dtype, device=cfg.DEVICE),
        )
        loss = loss_fct.forward(input=logits, target=labels)
        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
    ):
        outputs = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state  # [BATCH_SIZE, SEQ_LEN, 768]
        logits = self.ffwd.forward(logits)  # [BATCH_SIZE, SEQ_LEN, NUM_TAGS]

        loss = None
        if logits is not None:
            loss = self._compute_loss(
                logits=logits,
                labels=labels,
                num_tags=self.num_tags,
                loss_fct=self.loss_fct,
                attention_mask=attention_mask,
            )

        if return_dict:
            return {"logits": logits, "loss": loss}

        return BertEntityModelOutput(logits=logits, loss=loss)


# Train One Epoch
def train_one_epoch(
    model: BertEntityModel,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    dataloader: DataLoader,
    scaler: amp.grad_scaler.GradScaler,
    epoch: int,
) -> Tuple[float, float]:
    pbar = tqdm(enumerate(dataloader), desc="train ", total=len(dataloader))
    running_loss, dataset_size, running_hits = 0, 0, 0

    for step, batch in pbar:
        batch = {k: v.to(cfg.DEVICE) for k, v in batch.items()}

        with amp.autocast_mode.autocast():
            outputs = model.forward(**batch)
            logits, loss = outputs["logits"], outputs["loss"]
            del outputs

        scaler.scale(loss).backward()
        scaler.step(optimizer=optimizer)
        if scheduler is not None:
            scheduler.step()

        batch_size = batch["input_ids"].shape[0]
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        pbar.set_postfix(
            {"loss": f"{epoch_loss:.4f}"}
        )

        wandb.log(
            {
                "step": step,
                f"train/epoch_loss/epoch={epoch}": epoch_loss,
            }
        )

    return epoch_loss


# Valid One Epoch
@torch.no_grad()
def valid_one_epoch(
    model: BertEntityModel,
    dataloader: DataLoader,
    epoch: int,
) -> Tuple[float, float]:
    pbar = tqdm(enumerate(dataloader), desc="valid ", total=len(dataloader))
    running_loss, dataset_size, running_hits = 0, 0, 0

    for step, batch in pbar:
        batch = {k: v.to(cfg.DEVICE) for k, v in batch.items()}

        with amp.autocast_mode.autocast():
            outputs = model.forward(**batch)
            logits, loss = outputs["logits"], outputs["loss"]
            del outputs

        batch_size = batch["input_ids"].shape[0]
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        pbar.set_postfix(
            {"loss": f"{epoch_loss:.4f}"}
        )

        wandb.log(
            {
                "step": step,
                f"valid/epoch_loss/epoch={epoch}": epoch_loss,
            }
        )

    return epoch_loss


# Main Logic for training
def run_training(train_dataloader: DataLoader, valid_dataloader: DataLoader):
    run = wandb.init(
        project="legalease",
        notes="experiments for entity recognition from legal documents",
        tags=["nlp", "sih"],
        config={
            "epochs": cfg.EPOCHS,
            "learning_rate": cfg.LEARNING_RATE,
            "batch_size": cfg.BATCH_SIZE,
            "model": "Bert",
        },
        group=f"entity-extraction",
    )

    model = BertEntityModel(hidden_size=cfg.HIDDEN_SIZE, num_tags=cfg.NUM_TAGS).to(
        cfg.DEVICE
    )
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=30_000, eta_min=1e-5
    )
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.infty

    for epoch in range(cfg.EPOCHS):
        print("*" * 15)
        print(f"*** Epoch {epoch+1} ***")
        print("*" * 15)

        scaler = amp.grad_scaler.GradScaler()
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_dataloader,
            scaler=scaler,
            epoch=epoch,
        )

        valid_loss = valid_one_epoch(
            model=model,
            dataloader=valid_dataloader,
            epoch=epoch,
        )

        if valid_loss < best_loss:
            print(f"Loss improved from {best_loss} to {valid_loss}...")
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(best_model_wts, f"../../models/best_model_ner.pth")
            print(f"Best model checkpoints stored...")

            run.summary["BEST_TRAIN_LOSS"] = train_loss
            run.summary["BEST_VALID_LOSS"] = valid_loss

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "valid/loss": valid_loss,
            }
        )

    return model.load_state_dict(best_model_wts)


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    df = pd.read_csv("../data/ner/ner_train.csv")

    train_df, valid_df = df[:180035], df[180035:]
    train_dataset, valid_dataset = EntityRecognitionDataset(
        df=train_df, tokenizer=tokenizer
    ), EntityRecognitionDataset(df=valid_df, tokenizer=tokenizer)
    train_loader, valid_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False
    ), DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = run_training(train_dataloader=train_loader, valid_dataloader=valid_loader)
