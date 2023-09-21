import copy
from typing import Dict, List, Optional, Tuple

import einops
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn import model_selection, preprocessing
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast
from transformers import get_linear_schedule_with_warmup

# -------------------------------- Setup Wandb ------------------------------- #
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("WANDB")
    wandb.login(key=api_key)
    anonymous = None
except:
    anonymous = "must"
    print('To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')


# ---------------------------------- Config ---------------------------------- #
class cfg:
    batch_size: int = 16
    max_len: int = 128
    dropout: float = 0.1
    epochs: int = 10
    learning_rate: float = 3e-4
    hidden_size: int = 384
    entities: List[str] = [
        "WITNESS",
        "OTHER_PERSON",
        "STATUTE",
        "CASE_NUMBER",
        "GPE",
        "ORG",
        "DATE",
        "JUDGE",
        "PROVISION",
        "PETITIONER",
        "RESPONDENT",
        "COURT",
        "PRECEDENT",
    ]
    num_tags: int = len(entities)
    model_path: str = "bert-base-uncased"
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
        model_path)
    device: torch.device = torch.device(
        "cuda" if torch.has_cuda else "mps" if torch.has_mps else "cpu")


# ---------------------------------- Dataset --------------------------------- #
class LegalEntityDataset(Dataset):
    def __init__(self, texts: List[List[str]], tags: List[List[str]]) -> None:
        super().__init__()
        self.texts = texts
        self.tags = tags

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        text = self.texts[ix]
        tags = self.tags[ix]

        input_ids = []
        target_ids = []

        for i, s in enumerate(text):
            inputs = cfg.tokenizer.encode(text=s, add_special_tokens=False)
            input_ids.extend(inputs)
            target_ids.extend([tags[i]*len(inputs)])

        input_ids = input_ids[:cfg.max_len - 2]
        target_ids = target_ids[:cfg.max_len - 2]

        input_ids = [101] + input_ids + [102]
        target_ids = [0] + target_ids + [0]

        attention_mask = [1] * len(input_ids)
        token_type_ids = [1] * len(input_ids)

        padding_len = cfg.max_len - len(input_ids)

        input_ids = input_ids + ([0]*padding_len)
        attention_mask = attention_mask + ([0]*padding_len)
        token_type_ids = token_type_ids + ([0]*padding_len)
        target_ids = target_ids + ([0]*padding_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long)
        }


# ----------------------------------- Model ---------------------------------- #
class EntityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.distilbert = BertModel.from_pretrained(cfg.model_path)
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.ffwd = nn.Sequential(
            nn.Linear(in_features=768, out_features=cfg.hidden_size, bias=True),
            nn.SELU(),
            nn.Linear(in_features=cfg.hidden_size,
                      out_features=cfg.num_tags, bias=False)
        )

    @staticmethod
    def _compute_loss(logits: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        '''
        b - batch size
        s - sequence length
        c - num classes
        '''
        loss_fct = nn.CrossEntropyLoss()
        attention_mask = einops.rearrange(
            tensor=attention_mask, pattern='b s -> (b s)')
        logits = einops.rearrange(tensor=logits, pattern='b s c -> (b s) c')
        labels = einops.rearrange(tensor=labels, pattern='b s -> (b s)')
        labels = torch.where(
            attention_mask == 1,
            labels,
            torch.tensor(loss_fct.ignore_index,
                         dtype=labels.dtype, device=logits.device)
        )
        loss = loss_fct(logits, labels)
        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        out, _ = self.distilbert.forward(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.dropout(out)
        logits = self.ffwd(logits)

        loss = None
        if labels is not None:
            loss = self._compute_loss(
                logits=logits,
                attention_mask=attention_mask,
                labels=labels
            )

        return logits, loss


# ------------------------------ Train Function ------------------------------ #
def train_fn(model, optimizer, scheduler, dataloader, scaler, epoch) -> float:
    model.train()
    running_loss, dataset_size = 0, 0
    pbar = tqdm(enumerate(dataloader), desc='(train) ', total=len(dataloader))

    for _, batch in pbar:
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        with amp.autocast():
            _, loss = model.forward(**batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer=optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running_loss += (loss.item()*batch['input_ids'].shape[0])
        dataset_size += batch['input_ids'].shape[0]

        epoch_loss = running_loss / dataset_size
        pbar.set_postfix({'loss': f'{loss:.4f}'})

        wandb.log({f'train/epoch_loss/epoch={epoch}': epoch_loss})

    return epoch_loss


# ---------------------------- Validation Function --------------------------- #
@torch.no_grad()
def valid_fn(model, dataloader):
    model.eval()
    running_loss, dataset_size = 0, 0
    pbar = tqdm(enumerate(dataloader), desc='(valid) ', total=len(dataloader))

    for _, batch in pbar:
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        with amp.autocast():
            _, loss = model.forward(**batch)

        running_loss += (loss.item()*batch['input_ids'].shape[0])
        dataset_size += batch['input_ids'].shape[0]

        epoch_loss = running_loss / dataset_size
        pbar.set_postfix({'loss': f'{loss:.4f}'})

        wandb.log({f'valid/epoch_loss/epoch={epoch}': epoch_loss})

    return epoch_loss


# --------------------------- Process Data Function -------------------------- #
def process_data(data_file):
    df = pd.read_csv(data_file)

    encoder = preprocessing.LabelEncoder()

    df.loc[:, "tag"] = encoder.fit_transform(df["tag"])

    sentences = df.groupby("sentence #")["words"].apply(list).values
    tags = df.groupby("sentence #")['tag'].apply(list).values

    return sentences, tags, encoder


if __name__ == '__main__':
    sentences, tags, encoder = process_data('../data/ner/ner_train.csv')

    joblib.dump(encoder, "encoder.bin")

    num_tags = len(list(encoder.classes_))

    # build dataset
    (train_sentences, test_sentences, train_tags, test_tags) = model_selection.train_test_split(
        sentences, tags, random_state=42, test_size=0.2)

    train_dataset = LegalEntityDataset(texts=train_sentences, tags=train_tags)
    train_loader = DataLoader(
        dataset=train_dataset, shuffle=True, batch_size=cfg.batch_size, num_workers=4)

    valid_dataset = LegalEntityDataset(texts=test_sentences, tags=test_tags)
    valid_loader = DataLoader(
        dataset=valid_dataset, shuffle=True, batch_size=cfg.batch_size, num_workers=4)

    # define the model
    model = EntityModel()

    # build the optimizer
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())

    optimizer_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.001
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]

    num_training_steps = int(len(train_sentences) /
                             cfg.batch_size * cfg.epochs)
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=3e-4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_loss = np.infty

    run = wandb.init(
        project='legalease',
        notes="experiments for entity recognition from legal documents",
        tags=['sih', 'tags'],
        config={
            'epochs': cfg.epochs,
            'learning_rate': cfg.learning_rate,
            'batch_size': cfg.batch_size,
            'model': cfg.model_path,
            'hidden_size': cfg.hidden_size
        }
    )

    for epoch in range(cfg.epochs):
        print('-'*20)
        print(f'*** Epoch [{epoch+1}/{cfg.epochs}]')
        print('-'*20)

        train_loss = train_fn(model=model, optimizer=optimizer,
                              scheduler=scheduler, dataloader=train_loader)
        valid_loss = valid_fn(model=model, dataloader=valid_loader)

        if valid_loss < best_loss:
            print(
                f'Best loss decreased from {best_loss:.4f} to {valid_loss:.4f}')
            best_loss = valid_loss
            torch.save(model.state_dict(), 'best_model_ner.pt')
        
            run.summary['BEST_EPOCH'] = epoch
            run.summary['BEST_TRAIN_LOSS'] = train_loss
            run.summary['BEST_VALID_LOSS'] = valid_loss
        
        wandb.log({
            'train/loss': train_loss,
            'valid/loss': valid_loss
        })
    
    print(f"\t\t\t\t\t>>>>>>>> MODEL TRAINING DONE <<<<<<<<<")
