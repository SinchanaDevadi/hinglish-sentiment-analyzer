"""
train.py — Fine-tune mBERT on Hinglish sentiment data.

Dataset: SemEval-2020 Task 9 (Hinglish)
Download: https://ritual.uh.edu/semeval2020-task9/
Place as:  data/Hinglish_train.csv  &  data/Hinglish_dev.csv

Recommended: Run on Google Colab (free GPU)
    !python train.py
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm
from model_utils import get_tokenizer, get_model, MAX_LEN, LABELS

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH = "data/Hinglish_train.csv"
DEV_PATH   = "data/Hinglish_dev.csv"
SAVE_DIR   = "saved_model"
BATCH_SIZE = 16
EPOCHS     = 4
LR         = 2e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL2ID   = {l: i for i, l in enumerate(LABELS)}

# ── Dataset ───────────────────────────────────────────────────────────────────
class HinglishDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].map(LABEL2ID).tolist()
        self.tok    = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=MAX_LEN,
                       padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "sentence" in df.columns and "text" not in df.columns:
        df.rename(columns={"sentence": "text"}, inplace=True)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].str.strip().str.capitalize()
    return df[df["label"].isin(LABELS)]

# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    print(f"Device: {DEVICE}")
    tokenizer = get_tokenizer()
    model     = get_model(num_labels=3).to(DEVICE)

    train_df = load_data(TRAIN_PATH)
    dev_df   = load_data(DEV_PATH)
    print(f"Train: {len(train_df)} | Dev: {len(dev_df)}")

    train_loader = DataLoader(HinglishDataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(HinglishDataset(dev_df,   tokenizer), batch_size=BATCH_SIZE)

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_scheduler("linear", optimizer=optimizer,
                                num_warmup_steps=int(0.1 * total_steps),
                                num_training_steps=total_steps)

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(DEVICE),
                        attention_mask=batch["attention_mask"].to(DEVICE),
                        labels=batch["label"].to(DEVICE))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += out.loss.item()
        print(f"  Avg loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch} [eval]"):
                out = model(input_ids=batch["input_ids"].to(DEVICE),
                            attention_mask=batch["attention_mask"].to(DEVICE))
                preds.extend(out.logits.argmax(dim=1).cpu().numpy())
                labels.extend(batch["label"].numpy())

        report   = classification_report(labels, preds, target_names=LABELS, output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]
        print(classification_report(labels, preds, target_names=LABELS))

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  ✓ Best model saved — Macro F1: {best_f1:.4f}")

    print(f"\nDone! Best Macro F1: {best_f1:.4f} | Saved to: {SAVE_DIR}/")

if __name__ == "__main__":
    train()
