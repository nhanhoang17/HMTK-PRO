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
        print("Loc K-core (>= 10 unique interactions) ...")
        while True:
            u_unique = df.groupby("user_id")["track_id"].nunique()
            t_unique = df.groupby("track_id")["user_id"].nunique()
            valid_u  = u_unique[u_unique >= 10].index
            valid_t  = t_unique[t_unique >= 10].index
            df_new   = df[df["user_id"].isin(valid_u) &
                          df["track_id"].isin(valid_t)]
            if len(df_new) == len(df):
                break
            df = df_new.copy()

        print("Ma hoa IDs ...")
        df = df.copy()
        df["user_idx"]  = self.user_enc.fit_transform(df["user_id"])
        df["track_idx"] = self.track_enc.fit_transform(df["track_id"])
        df["tz_idx"]    = self.tz_enc.fit_transform(df["time_zone"])
        df["lang_idx"]  = self.lang_enc.fit_transform(df["lang"])
        df["tlang_idx"] = self.tlang_enc.fit_transform(df["tweet_lang"])

        print(f"\nChia Train/Val/Test theo USER-LEVEL voi ti le {Config.USER_SPLIT_RATIO} ...")
        unique_users = df["user_idx"].unique()
        np.random.seed(42)
        np.random.shuffle(unique_users)

        n_users = len(unique_users)
        n_train = int(n_users * Config.USER_SPLIT_RATIO[0])
        n_val   = int(n_users * Config.USER_SPLIT_RATIO[1])

        train_users = unique_users[:n_train]
        val_users   = unique_users[n_train : n_train + n_val]
        test_users  = unique_users[n_train + n_val:]

        # 1. Xac dinh item can predict trong tap Val va Test
        val_df  = df[df["user_idx"].isin(val_users)].groupby("user_idx").tail(1).copy()
        test_df = df[df["user_idx"].isin(test_users)].groupby("user_idx").tail(1).copy()

        # 2. Xac dinh user nao la cold-start trong tap VAL va TEST
        n_val_cold = int(len(val_users) * Config.COLD_START_RATIO)
        val_cold_users = val_users[:n_val_cold]
        val_warm_users = val_users[n_val_cold:]

        n_test_cold = int(len(test_users) * Config.COLD_START_RATIO)
        test_cold_users = test_users[:n_test_cold]
        test_warm_users = test_users[n_test_cold:]

        # Gom tat ca cold-start user (ca val va test) de xoa lich su
        all_cold_users = np.concatenate([val_cold_users, test_cold_users])

        # 3. Xay dung Interaction Matrix Base (De tinh su_vector va SVD)
        # BỎ ĐI các item mục tiêu và BỎ SẠCH lịch sử của ALL cold_start
        drop_indices = pd.concat([
            val_df,
            test_df,
            df[df["user_idx"].isin(all_cold_users)]
        ]).index
        interaction_df = df.drop(index=drop_indices).copy()

        # 4. Trích xuất profile gốc dùng chung để nội suy Z
        all_users_profile_df = df.drop_duplicates(subset=["user_idx"]).copy()

        print(f"  Train Users (70%): {len(train_users)} | Val Users (10%): {len(val_users)} | Test Users (20%): {len(test_users)}")
        print(f"  Trong nhom Val:  COLD-START = {len(val_cold_users)} | WARM = {len(val_warm_users)}")
        print(f"  Trong nhom Test: COLD-START = {len(test_cold_users)} | WARM = {len(test_warm_users)}")
        
        return interaction_df, val_df, test_df, train_users, val_users, test_users, val_cold_users, test_cold_users, all_users_profile_df

    def build_interaction_matrix(self, interaction_df):
        print("Xay dung sparse matrix ...")
        n_users = len(self.user_enc.classes_)
        n_items = len(self.track_enc.classes_)
        agg = (interaction_df.groupby(["user_idx", "track_idx"])
                             .size().reset_index(name="count"))
        data = np.log1p(agg["count"].values).astype(np.float32)
        return csr_matrix(
            (data, (agg["user_idx"].values, agg["track_idx"].values)),
            shape=(n_users, n_items))

    @property
    def n_users(self): return len(self.user_enc.classes_)

    @property
    def n_items(self):  return len(self.track_enc.classes_)
