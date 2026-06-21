import torch

class Config:
    # --- Data & Paths ---
    DATA_PATH        = "context_content_features.csv"
    FM_WEIGHTS_PATH  = "fm_trained_real.pth"

    # --- Dimensions (paper Sect. 4.2) ---
    M_DIM        = 128   # Chieu item embedding (SVD)
    K_DIM        = 64    # Chieu latent space Z
    FM_EMBED_DIM = 16    # Chieu embedding moi feature trong FM
    # Profile dim = 3 * FM_EMBED_DIM = 48

    # --- Training CFLS (paper Sect. 4.2) ---
    BATCH_SIZE  = 64      # paper: mini-batch = 64
    EPOCHS_CFLS = 10    # paper: epochs = 10
    PATIENCE    = 3      # paper: early stopping patience = 3
    LR          = 0.001   # paper: Adam lr = 0.001

    # --- Training FM ---
    EPOCHS_FM     = 5
    LR_FM         = 0.001
    BATCH_SIZE_FM = 2048

    # --- Evaluation ---
    COLD_START_RATIO = 0.7
    TOP_K            = [20, 40, 60, 80]
    SUPPORT_SIZE     = 6000   # So train users dua vao GP (tranh OOM)
    N_SAMPLES        = 5      # So lan sample GP posterior
    EVAL_BATCH_SIZE  = 512    # Batch size khi tinh top-K scores

    # --- Device ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
