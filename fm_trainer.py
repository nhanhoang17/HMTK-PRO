"""
fm_trainer.py
-------------
Train FM de hoc dense user profile xu.
Negative sampling: random ca 3 fields tu toan bo pool, kiem tra collision.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from config import Config
from data_prep import DataPipeline
from embeddings import FactorizationMachine


def prepare_fm_data(train_df):
    """
    Positive: (tz, lang, tlang) thuc su cua user.
    Negative: random ca 3 fields tu pool toan bo (tranh noisy label).
    """
    print("Chuan bi du lieu FM ...")
    profiles = (
        train_df.drop_duplicates(subset=["user_idx"])
                [["tz_idx", "lang_idx", "tlang_idx"]]
                .values.astype(np.int64)
    )
    pos_tz, pos_lang, pos_tlang = profiles[:,0], profiles[:,1], profiles[:,2]
    n   = len(profiles)
    rng = np.random.default_rng(42)

    neg_tz    = rng.choice(pos_tz,    size=n, replace=True)
    neg_lang  = rng.choice(pos_lang,  size=n, replace=True)
    neg_tlang = rng.choice(pos_tlang, size=n, replace=True)

    pos_set = set(zip(pos_tz.tolist(), pos_lang.tolist(), pos_tlang.tolist()))
    while True:
        mask  = np.array([(t,l,tl) in pos_set
                          for t,l,tl in zip(neg_tz, neg_lang, neg_tlang)])
        n_col = mask.sum()
        if n_col == 0:
            break
        neg_tz[mask]    = rng.choice(pos_tz,    size=n_col, replace=True)
        neg_lang[mask]  = rng.choice(pos_lang,  size=n_col, replace=True)
        neg_tlang[mask] = rng.choice(pos_tlang, size=n_col, replace=True)

    idx = rng.permutation(2 * n)
    ds  = TensorDataset(
        torch.tensor(np.concatenate([pos_tz,    neg_tz])[idx],    dtype=torch.long),
        torch.tensor(np.concatenate([pos_lang,  neg_lang])[idx],  dtype=torch.long),
        torch.tensor(np.concatenate([pos_tlang, neg_tlang])[idx], dtype=torch.long),
        torch.tensor(np.concatenate([np.ones(n), np.zeros(n)])[idx], dtype=torch.float32),
    )
    print(f"  FM data: {len(ds)} rows (pos={n}, neg={n})")
    return DataLoader(ds, batch_size=Config.BATCH_SIZE_FM, shuffle=True)


def train_fm():
    print(f"=== HUAN LUYEN FM (Device: {Config.DEVICE}) ===")
    pipeline = DataPipeline()
    df       = pipeline.load()
    train_df, _, _ = pipeline.process_and_split(df)

    dl = prepare_fm_data(train_df)
    model = FactorizationMachine(
        len(pipeline.tz_enc.classes_),
        len(pipeline.lang_enc.classes_),
        len(pipeline.tlang_enc.classes_),
        Config.FM_EMBED_DIM,
    ).to(Config.DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR_FM)
    best_loss = float("inf")

    for epoch in range(Config.EPOCHS_FM):
        model.train()
        total = 0.0
        for tz, lang, tlang, y in dl:
            tz, lang, tlang, y = (x.to(Config.DEVICE) for x in [tz,lang,tlang,y])
            optimizer.zero_grad()
            loss = criterion(model(tz, lang, tlang).squeeze(), y)
            loss.backward(); optimizer.step()
            total += loss.item()
        avg = total / len(dl)
        print(f"  FM Epoch [{epoch+1}/{Config.EPOCHS_FM}] loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), Config.FM_WEIGHTS_PATH)

    print(f"Da luu FM: {Config.FM_WEIGHTS_PATH} (best={best_loss:.4f})")


if __name__ == "__main__":
    train_fm()
