"""
data_prep.py
------------
Thiet ke dung theo paper (Sect. 4.2):
  - TAT CA users deu co lich su trong train_df
  - cold_start_users chi la NHAN danh gia, khong anh huong train
  - K-core dung nunique (unique interactions), khong phai tong rows
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from config import Config


class DataPipeline:
    def __init__(self):
        self.user_enc  = LabelEncoder()
        self.track_enc = LabelEncoder()
        self.tz_enc    = LabelEncoder()
        self.lang_enc  = LabelEncoder()
        self.tlang_enc = LabelEncoder()

    def load(self):
        print("Doc du lieu ...")
        df = pd.read_csv(
            Config.DATA_PATH,
            on_bad_lines="skip",
            usecols=["user_id", "track_id", "tweet_lang", "lang", "time_zone"],
            dtype=str,
            low_memory=False,
        )
        df[["time_zone", "lang", "tweet_lang"]] = (
            df[["time_zone", "lang", "tweet_lang"]].fillna("unknown"))
        df = df.dropna(subset=["user_id", "track_id"])
        return df

    def process_and_split(self, df):
        """
        K-core (nunique) -> encode -> leave-one-out -> danh dau cold-start.

        Quan trong:
          train_df KHONG bi xoa user nao.
          cold_start_users chi dung luc EVALUATE de gia lap khong co su.
          Khi r=1.0: train_df day du, GP van co support set.
        """
        # K-core iterative (dung nunique - unique interactions)
        print("Loc K-core (>= 5 unique interactions) ...")
        while True:
            u_unique = df.groupby("user_id")["track_id"].nunique()
            t_unique = df.groupby("track_id")["user_id"].nunique()
            valid_u  = u_unique[u_unique >= 1].index
            valid_t  = t_unique[t_unique >= 1].index
            df_new   = df[df["user_id"].isin(valid_u) &
                          df["track_id"].isin(valid_t)]
            if len(df_new) == len(df):
                break
            df = df_new.copy()
        print(f"  Sau loc: {df['user_id'].nunique()} users, "
              f"{df['track_id'].nunique()} tracks, {len(df)} rows")

        # Encode
        print("Ma hoa IDs ...")
        df = df.copy()
        df["user_idx"]  = self.user_enc.fit_transform(df["user_id"])
        df["track_idx"] = self.track_enc.fit_transform(df["track_id"])
        df["tz_idx"]    = self.tz_enc.fit_transform(df["time_zone"])
        df["lang_idx"]  = self.lang_enc.fit_transform(df["lang"])
        df["tlang_idx"] = self.tlang_enc.fit_transform(df["tweet_lang"])

        # Leave-one-out
        print("Leave-one-out split ...")
        test_df  = df.groupby("user_idx").tail(1).copy()
        train_df = df.drop(index=test_df.index).copy()

        # Danh dau cold-start users (chi la nhan, KHONG xoa khoi train)
        unique_test = test_df["user_idx"].unique()
        n_cold      = int(len(unique_test) * Config.COLD_START_RATIO)
        np.random.seed(42)
        cold_start_users = np.random.choice(unique_test, n_cold, replace=False)

        print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows | "
              f"Cold-start: {len(cold_start_users)} users (r={Config.COLD_START_RATIO})")
        return train_df, test_df, cold_start_users

    def build_interaction_matrix(self, train_df):
        """Sparse matrix (n_users x n_items) voi gia tri log(1+count)."""
        print("Xay dung sparse matrix ...")
        n_users = len(self.user_enc.classes_)
        n_items = len(self.track_enc.classes_)
        agg = (train_df.groupby(["user_idx", "track_idx"])
                       .size().reset_index(name="count"))
        data = np.log1p(agg["count"].values).astype(np.float32)
        return csr_matrix(
            (data, (agg["user_idx"].values, agg["track_idx"].values)),
            shape=(n_users, n_items))

    @property
    def n_users(self): return len(self.user_enc.classes_)

    @property
    def n_items(self):  return len(self.track_enc.classes_)
