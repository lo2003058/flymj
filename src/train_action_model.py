"""Phase 2 Part 4：訓練 multi-head ActionNet（Arm A：真 connectome mask）。

用 action_features_scaled.npz（2009-2018 十年，見 build_action_dataset_scaled.py
/build_action_features_scaled.py）訓練，唔再用原本嗰 2009 年單年版本——
train_scaling_experiment.py 已經證實過同一批擴大嘅牌譜可以將純掉牌 model
嘅 test accuracy 由 65.7% 谷到 69.7%，冇理由呢個部署緊、game_app.py 實際
用嚟做叫牌/立直/自摸/防守建議嘅 model 淨係用返細嗰份 dataset。

跑法： uv run python src/train_action_model.py
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from action_model import REACTION_ACTION_TYPES, SELF_ACTION_TYPES, ActionNet

FEATURES_PATH = "data/processed/action_features_scaled.npz"
MASKS_PATH = "data/processed/masks.npz"
OUT_PATH = "data/processed/action_model_arm_a.pt"

MAX_EPOCHS = 20
PATIENCE = 3
BATCH_SIZE = 1024
LR = 1e-3
SEED = 0


def make_loader(
    x: np.ndarray, self_type: np.ndarray, discard_tile: np.ndarray, reaction: np.ndarray,
    batch_size: int, shuffle: bool, seed: int | None = None,
) -> DataLoader:
    tensors = (
        torch.from_numpy(x),
        torch.from_numpy(self_type.astype(np.int64)),
        torch.from_numpy(discard_tile.astype(np.int64)),
        torch.from_numpy(reaction.astype(np.int64)),
    )
    dataset = TensorDataset(*tensors)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def compute_loss(outputs, self_type, discard_tile, reaction, loss_fn) -> torch.Tensor:
    total = outputs["discard"].new_zeros(())
    n_terms = 0

    mask = self_type >= 0
    if mask.any():
        total = total + loss_fn(outputs["self_type"][mask], self_type[mask])
        n_terms += 1

    mask = discard_tile >= 0
    if mask.any():
        total = total + loss_fn(outputs["discard"][mask], discard_tile[mask])
        n_terms += 1

    mask = reaction >= 0
    if mask.any():
        total = total + loss_fn(outputs["reaction"][mask], reaction[mask])
        n_terms += 1

    return total / max(n_terms, 1)


@torch.no_grad()
def evaluate(model, x, self_type, discard_tile, reaction, device, batch_size):
    model.eval()
    loader = make_loader(x, self_type, discard_tile, reaction, batch_size, shuffle=False)
    correct = {"self_type": 0, "discard": 0, "reaction": 0}
    total = {"self_type": 0, "discard": 0, "reaction": 0}

    for xb, st, dt, rc in loader:
        xb = xb.float().to(device)
        st, dt, rc = st.to(device), dt.to(device), rc.to(device)
        out = model(xb)

        m = st >= 0
        if m.any():
            correct["self_type"] += (out["self_type"][m].argmax(dim=1) == st[m]).sum().item()
            total["self_type"] += int(m.sum().item())
        m = dt >= 0
        if m.any():
            correct["discard"] += (out["discard"][m].argmax(dim=1) == dt[m]).sum().item()
            total["discard"] += int(m.sum().item())
        m = rc >= 0
        if m.any():
            correct["reaction"] += (out["reaction"][m].argmax(dim=1) == rc[m]).sum().item()
            total["reaction"] += int(m.sum().item())

    accs = {k: (correct[k] / total[k] if total[k] else float("nan")) for k in correct}
    overall = sum(correct.values()) / sum(total.values())
    return accs, overall


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    feats = np.load(FEATURES_PATH)
    masks = np.load(MASKS_PATH)

    x_full = feats["X"]
    self_type_full = feats["self_type"]
    discard_tile_full = feats["discard_tile"]
    reaction_full = feats["reaction"]
    split_full = feats["split"]

    data = {}
    for split in ("train", "val", "test"):
        m = split_full == split
        data[split] = (x_full[m], self_type_full[m], discard_tile_full[m], reaction_full[m])
        print(f"{split}: {int(m.sum())} 行")

    torch.manual_seed(SEED)
    mask_pn_kc = torch.tensor(masks["mask_pn_kc_real"], dtype=torch.float32, device=device)
    mask_kc_mbon = torch.tensor(masks["mask_kc_mbon_real"], dtype=torch.float32, device=device)

    n_channels, n_tiles = x_full.shape[1], x_full.shape[2]
    model = ActionNet(n_channels, n_tiles, mask_pn_kc, mask_kc_mbon).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model 參數量: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    x_train, st_train, dt_train, rc_train = data["train"]
    train_loader = make_loader(x_train, st_train, dt_train, rc_train, BATCH_SIZE, shuffle=True, seed=SEED)

    best_val = -1.0
    best_state = None
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, st, dt, rc in train_loader:
            xb = xb.float().to(device)
            st, dt, rc = st.to(device), dt.to(device), rc.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = compute_loss(out, st, dt, rc, loss_fn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_accs, val_overall = evaluate(model, *data["val"], device, BATCH_SIZE)
        print(
            f"epoch {epoch + 1}/{MAX_EPOCHS} loss={total_loss / n_batches:.4f} "
            f"val: self_type={val_accs['self_type']:.4f} discard={val_accs['discard']:.4f} "
            f"reaction={val_accs['reaction']:.4f} overall={val_overall:.4f}"
        )

        if val_overall > best_val:
            best_val = val_overall
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"early stop at epoch {epoch + 1}（best val_overall={best_val:.4f} @ epoch {best_epoch + 1}）")
                break

    model.load_state_dict(best_state)
    test_accs, test_overall = evaluate(model, *data["test"], device, BATCH_SIZE)
    print(
        f"\n最終 test accuracy: self_type={test_accs['self_type']:.4f} discard={test_accs['discard']:.4f} "
        f"reaction={test_accs['reaction']:.4f} overall={test_overall:.4f}"
    )

    torch.save(
        {"model_state": model.state_dict(), "n_channels": n_channels, "n_tiles": n_tiles},
        OUT_PATH,
    )
    print(f"已存 {OUT_PATH}")


if __name__ == "__main__":
    main()
