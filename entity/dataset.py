from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import config as cfg


class EntityRecognitionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizerFast) -> None:
        super().__init__()
        self.df = df
        self.texts = self.df.groupby("sentence #")["words"].apply(list).values
        self.tags = self.df.groupby("sentence #")["tag"].apply(list).values
        self.tokenizer = tokenizer
        self.inv_mappings = dict(enumerate(list(set(self.df["tag"].tolist()))))
        self.mappings = {v: k for k, v in self.inv_mappings.items()}

        # print(self.tokenizer.pad_token_id, self.tokenizer.pad_token)
        # print(self.texts[:3])

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        text = self.texts[ix]
        tags = self.tags[ix]
        encodings, attention_masks, labels = [], [], []

        for i, w in enumerate(text):
            tag = tags[i]
            tag = self.mappings[tag]
            try:
                enc = self.tokenizer.encode(w, add_special_tokens=False)
            except:
                enc = []
            encodings.extend(enc)
            labels.extend([tag] * len(enc))

        encodings = [101] + encodings[: cfg.MAX_LEN - 2] + [102]
        labels = [0] + labels[: cfg.MAX_LEN - 2] + [0]
        attention_masks = [1] * len(encodings)
        
        encodings = encodings + [0]*(cfg.MAX_LEN - len(encodings))
        labels = labels + [-100]*(cfg.MAX_LEN - len(labels))
        attention_masks = attention_masks  + [1]*(cfg.MAX_LEN - len(attention_masks))

        return {
            "input_ids": torch.tensor(
                encodings, dtype=torch.long
            ),  # (BATCH_SIZE, SEQ_LEN)
            "attention_mask": torch.tensor(
                attention_masks, dtype=torch.long
            ),  # (BATCH_SIZE, SEQ_LEN)
            "labels": torch.tensor(labels, dtype=torch.long),  # (BATCH_SIZE, SEQ_LEN)
        }

