"""Phase 1 Part 3：核心推理邏輯 —— 讀入手動輸入嘅場面，攞返個 model
對手牌入面每隻牌嘅評分，同 PN/KC/MBON 三層嘅 activation（俾 UI 畫圖用）。
"""

from dataclasses import dataclass

import numpy as np
import torch

from features import encode
from model import MahjongNet
from paifu_replay import FullDiscardState

N_TILE_TYPES = 34
N_PLAYERS = 4


def load_model(checkpoint_path: str, masks_npz, device: torch.device) -> MahjongNet:
    # weights_only=False：呢個 checkpoint 淨係由 train_and_save_model.py 呢個
    # 本地 script 產生（入面有 pn_ids/kc_ids/mbon_ids 呢啲 numpy array，唔喺
    # weights_only allowlist 度），唔係下載返嚟嘅唔信任檔案。
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mask_pn_kc = torch.tensor(masks_npz["mask_pn_kc_real"], dtype=torch.float32, device=device)
    mask_kc_mbon = torch.tensor(masks_npz["mask_kc_mbon_real"], dtype=torch.float32, device=device)
    model = MahjongNet(ckpt["n_channels"], ckpt["n_tiles"], mask_pn_kc, mask_kc_mbon).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def build_state(
    hand_counts: list[int],
    hand_red_counts: list[int] | None = None,
    meld_counts: list[list[int]] | None = None,
    discard_counts: list[list[int]] | None = None,
    dora_tiles: list[int] | None = None,
    round_wind: int = 0,
    seat_wind: int = 0,
    riichi: list[bool] | None = None,
    seat: int = 0,
) -> FullDiscardState:
    """由人手輸入嘅場面資訊，砌一個可以直接餵俾 features.encode() 嘅 state。"""
    return FullDiscardState(
        event_index=0,
        seat=seat,
        hand_counts=list(hand_counts),
        hand_red_counts=list(hand_red_counts) if hand_red_counts else [0] * N_TILE_TYPES,
        meld_counts=meld_counts or [[0] * N_TILE_TYPES for _ in range(N_PLAYERS)],
        discard_counts=discard_counts or [[0] * N_TILE_TYPES for _ in range(N_PLAYERS)],
        riichi=riichi or [False] * N_PLAYERS,
        dora_tiles=dora_tiles or [],
        round_wind=round_wind,
        seat_wind=seat_wind,
        discard_tile=0,
        discard_is_red=False,
        is_riichi=False,
        is_tsumogiri=False,
    )


@dataclass
class PredictionResult:
    probs: np.ndarray  # (34,) softmax，包括手牌冇嘅牌（僅供參考）
    ranked_hand_tiles: list[tuple[int, float]]  # 手牌入面實際有嘅牌，按分數排序 (tile_kind, prob)
    activations: dict[str, np.ndarray]  # "pn"/"kc"/"mbon" -> 1D array


def predict(model: MahjongNet, state: FullDiscardState, device: torch.device) -> PredictionResult:
    x = encode(state)
    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, acts = model.forward_with_activations(x_t)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    hand_tiles = [kind for kind, count in enumerate(state.hand_counts) if count > 0]
    ranked = sorted(((kind, float(probs[kind])) for kind in hand_tiles), key=lambda kv: kv[1], reverse=True)

    activations = {name: act.squeeze(0).cpu().numpy() for name, act in acts.items()}
    return PredictionResult(probs=probs, ranked_hand_tiles=ranked, activations=activations)
