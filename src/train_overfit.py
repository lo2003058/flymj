"""Step 5: an overfit test on 10,000 samples, to verify the whole pipeline
(mask -> model -> training) has no bugs.

Only tests Arm A (the real connectome mask). The goal here is "does the
pipeline work at all," not "which arm is best" (that comparison happens in
Step 7). If training on 10,000 samples doesn't reach close to 100%
training accuracy, the pipeline has a bug (mask wired wrong, data
misaligned, gradients not flowing, etc.).

Run: uv run python src/train_overfit.py
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
    print(f"features.npz has {len(y_all)} samples total")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_all), size=min(N_SAMPLES, len(y_all)), replace=False)
    x = torch.tensor(x_all[idx], dtype=torch.float32, device=device)
    y = torch.tensor(y_all[idx], dtype=torch.long, device=device)
    print(f"Selected {len(idx)} samples for the overfit test")

    mask_pn_kc = torch.tensor(masks["mask_pn_kc_real"], dtype=torch.float32, device=device)
    mask_kc_mbon = torch.tensor(masks["mask_kc_mbon_real"], dtype=torch.float32, device=device)
    print(f"mask_pn_kc shape={tuple(mask_pn_kc.shape)}  mask_kc_mbon shape={tuple(mask_kc_mbon.shape)}")

    n_channels, n_tiles = x.shape[1], x.shape[2]
    model = MahjongNet(n_channels, n_tiles, mask_pn_kc, mask_kc_mbon).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model total parameters: {n_params:,}")

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
    print(f"\nFinal training accuracy: {final_acc:.4f}")
    if final_acc < 0.9:
        print("Warning: didn't overfit to near 100% — the pipeline may have a bug, investigate")
    else:
        print("Overfit succeeded, no obvious pipeline bug")


if __name__ == "__main__":
    main()
