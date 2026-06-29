import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data_prep import DataPipeline
from embeddings import FactorizationMachine


def _make_dataloader(interaction_df, user_subset):
    """
    FM regression trên observed interactions (pointwise, không negative sampling).

    Mỗi sample: (tz, lang, tlang, track_idx, log1p_count)
    FM học predict implicit score cho các cặp (profile, item) đã quan sát.
    Không cần negative sampling vì FM là regression model, không phải ranking model.
    Missing entries được bỏ qua hoàn toàn (không gán score = 0).
    """
    df = interaction_df[interaction_df["user_idx"].isin(user_subset)].copy()
    if len(df) == 0:
        return None

    agg = (df.groupby(["track_idx", "tz_idx", "lang_idx", "tlang_idx"])
             .size().reset_index(name="count"))
    agg["score"] = np.log1p(agg["count"].values).astype(np.float32)

    ds = TensorDataset(
        torch.tensor(agg["tz_idx"].values,    dtype=torch.long),
        torch.tensor(agg["lang_idx"].values,  dtype=torch.long),
        torch.tensor(agg["tlang_idx"].values, dtype=torch.long),
        torch.tensor(agg["track_idx"].values, dtype=torch.long),
        torch.tensor(agg["score"].values,     dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=Config.BATCH_SIZE_FM, shuffle=True)


def train_fm():
    print(f"=== HUAN LUYEN FM (Device: {Config.DEVICE}) ===")
    pipeline = DataPipeline()
    df       = pipeline.load()
    interaction_df, _, _, train_users, val_users, *_ = pipeline.process_and_split(df)

    train_dl = _make_dataloader(interaction_df, train_users)
    val_dl   = _make_dataloader(interaction_df, val_users)

    model = FactorizationMachine(
        num_tz    = len(pipeline.tz_enc.classes_),
        num_lang  = len(pipeline.lang_enc.classes_),
        num_tlang = len(pipeline.tlang_enc.classes_),
        num_items = pipeline.n_items,
        embed_dim = Config.FM_EMBED_DIM,
    ).to(Config.DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR_FM)
    best_val, best_state = float("inf"), None

    for epoch in range(Config.EPOCHS_FM):
        model.train()
        train_loss = 0.0
        for tz, lang, tlang, item, y in train_dl:
            tz, lang, tlang, item, y = (x.to(Config.DEVICE)
                                        for x in [tz, lang, tlang, item, y])
            optimizer.zero_grad()
            loss = criterion(model(tz, lang, tlang, item).squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tz, lang, tlang, item, y in val_dl:
                tz, lang, tlang, item, y = (x.to(Config.DEVICE)
                                            for x in [tz, lang, tlang, item, y])
                val_loss += criterion(
                    model(tz, lang, tlang, item).squeeze(), y).item()
        val_loss /= len(val_dl)

        print(f"  FM Epoch [{epoch+1}/{Config.EPOCHS_FM}] "
              f"train_mse={train_loss:.4f} | val_mse={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, Config.FM_WEIGHTS_PATH)
    print(f"Da luu FM: {Config.FM_WEIGHTS_PATH} (best_val_mse={best_val:.4f})")


if __name__ == "__main__":
    train_fm()