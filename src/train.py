"""Step 7：訓練一個 arm 嘅可重用邏輯（minibatch，data 留喺 CPU 用 uint8，
一個 batch 先轉 float32 搬落 device——全量 data 一次過搬落 MPS 會爆記憶體）。
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from arms import build_arm_masks
from model import MahjongNet


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int | None = None) -> DataLoader:
    x_t = torch.from_numpy(X)  # uint8, (N, C, 34)
    y_t = torch.from_numpy(y.astype(np.int64))
    dataset = TensorDataset(x_t, y_t)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int) -> float:
    model.eval()
    loader = make_loader(X, y, batch_size, shuffle=False)
    correct = 0
    n = 0
    for xb, yb in loader:
        xb = xb.float().to(device)
        yb = yb.to(device)
        logits = model(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += len(yb)
    return correct / n


def train_one_arm(
    arm: str,
    seed: int,
    masks_npz,
    data: dict[str, np.ndarray],
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    lr: float,
    patience: int = 3,
    verbose_tag: str = "",
) -> dict:
    """訓練到 max_epochs，或者 val_acc 連續 `patience` 個 epoch 冇再創新高就
    早停。最終用返 val_acc 最好嗰個 epoch 嘅 weight 嚟 evaluate test set，
    咁樣每條 arm 都訓練到佢自己嘅上限先比較，唔會因為固定 epoch 數而偏袒
    邊一條 arm（例如收斂快啲嘅 arm）。
    """
    torch.manual_seed(seed)

    mask_pn_kc, mask_kc_mbon = build_arm_masks(arm, seed, masks_npz)
    mask_pn_kc = mask_pn_kc.to(device)
    mask_kc_mbon = mask_kc_mbon.to(device)

    n_channels, n_tiles = data["X_train"].shape[1], data["X_train"].shape[2]
    model = MahjongNet(n_channels, n_tiles, mask_pn_kc, mask_kc_mbon).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = make_loader(data["X_train"], data["y_train"], batch_size, shuffle=True, seed=seed)

    history = []
    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        for xb, yb in train_loader:
            xb = xb.float().to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_n += len(yb)

        train_loss = total_loss / total_n
        train_acc = total_correct / total_n
        val_acc = evaluate(model, data["X_val"], data["y_val"], device, batch_size)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(
            f"[{verbose_tag or arm} seed={seed}] epoch {epoch + 1}/{max_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"[{verbose_tag or arm} seed={seed}] early stop at epoch {epoch + 1} "
                    f"(best val_acc={best_val_acc:.4f} @ epoch {best_epoch + 1})"
                )
                break

    model.load_state_dict(best_state)
    test_acc = evaluate(model, data["X_test"], data["y_test"], device, batch_size)
    return {
        "n_params": n_params,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "model": model,
    }
