"""Step 5：用 1 萬個 sample 做 overfit test，驗證成條 pipeline（mask -> model
-> training）冇 bug。

淨係用 Arm A（真 connectome mask）測試。呢步嘅目的係「pipeline work 唔 work」，
唔係「三條 arm 邊條好」（三條 arm 對比係 Step 7 先做）。如果 1 萬個 sample
訓練唔到接近 100% training accuracy，即係 pipeline 有 bug（mask 駁錯、
data 冇對齊、gradient 冇流過等等）。

跑法： uv run python src/train_overfit.py
"""

import numpy as np
import torch
from torch import nn

from model import MahjongNet

FEATURES_PATH = "data/processed/features.npz"
MASKS_PATH = "data/processed/masks.npz"
N_SAMPLES = 10_000
N_EPOCHS = 300
LR = 1e-3


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    feats = np.load(FEATURES_PATH)
    masks = np.load(MASKS_PATH)

    x_all = feats["X"]
    y_all = feats["y"]
    print(f"features.npz 總共 {len(y_all)} 個樣本")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_all), size=min(N_SAMPLES, len(y_all)), replace=False)
    x = torch.tensor(x_all[idx], dtype=torch.float32, device=device)
    y = torch.tensor(y_all[idx], dtype=torch.long, device=device)
    print(f"揀咗 {len(idx)} 個 sample 做 overfit test")

    mask_pn_kc = torch.tensor(masks["mask_pn_kc_real"], dtype=torch.float32, device=device)
    mask_kc_mbon = torch.tensor(masks["mask_kc_mbon_real"], dtype=torch.float32, device=device)
    print(f"mask_pn_kc shape={tuple(mask_pn_kc.shape)}  mask_kc_mbon shape={tuple(mask_kc_mbon.shape)}")

    n_channels, n_tiles = x.shape[1], x.shape[2]
    model = MahjongNet(n_channels, n_tiles, mask_pn_kc, mask_kc_mbon).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model 總參數量: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % 20 == 0:
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"epoch {epoch + 1:4d}  loss={loss.item():.4f}  train_acc={acc:.4f}")

    with torch.no_grad():
        final_logits = model(x)
        final_acc = (final_logits.argmax(dim=1) == y).float().mean().item()
    print(f"\n最終 training accuracy: {final_acc:.4f}")
    if final_acc < 0.9:
        print("警告：冇 overfit 到接近 100%，pipeline 可能有 bug，要查")
    else:
        print("Overfit 成功，pipeline 冇明顯 bug")


if __name__ == "__main__":
    main()
