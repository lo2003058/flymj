"""麻雀 discard prediction 嘅網絡架構（跟 data/doc/task.md 嗰個圖）：

  麻雀 feature (C x 34)
    -> Conv1d 前端
    -> Linear -> PN 層
    -> MaskedLinear(PN -> KC)
    -> ReLU
    -> MaskedLinear(KC -> MBON)
    -> Linear -> 34 logits

淨係喺兩個 MaskedLinear 之間有 ReLU，係跟返 task.md 個圖嘅字面意思
（Conv1d 前端入面用幾多層、有冇 ReLU 就係呢個 module 自己嘅實作細節，
 冇喺個圖度講明，跟業界慣例加）。
"""

import torch
import torch.nn.functional as F
from torch import nn


class MaskedLinear(nn.Module):
    """nn.Linear，但 weight 入面對唔上 connectome mask 嘅位永遠係 0。

    Mask 係 buffer（唔係 parameter），唔會被訓練或者 optimizer 更新。
    """

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super().__init__()
        if mask.shape != (out_features, in_features):
            raise ValueError(f"mask shape {tuple(mask.shape)} 同 (out_features, in_features)="
                              f"{(out_features, in_features)} 唔夾")
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("mask", mask.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)


class MahjongNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_tiles: int,
        mask_pn_kc: torch.Tensor,
        mask_kc_mbon: torch.Tensor,
        conv_channels: int = 64,
    ):
        super().__init__()
        n_kc, n_pn = mask_pn_kc.shape
        n_mbon, n_kc2 = mask_kc_mbon.shape
        if n_kc != n_kc2:
            raise ValueError(f"mask_pn_kc 嘅 KC 維度 {n_kc} 同 mask_kc_mbon 嘅 KC 維度 {n_kc2} 唔夾")

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.to_pn = nn.Linear(conv_channels * n_tiles, n_pn)
        self.pn_to_kc = MaskedLinear(n_pn, n_kc, mask_pn_kc)
        self.kc_to_mbon = MaskedLinear(n_kc, n_mbon, mask_kc_mbon)
        self.to_logits = nn.Linear(n_mbon, n_tiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_activations(x)[0]

    def forward_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """同 forward 一樣，但連 PN/KC/MBON 三層嘅中間輸出都一齊回傳，
        俾 UI 畫「邊粒神經元依家活躍」用。"""
        h = self.conv(x)  # (B, conv_channels, n_tiles)
        h = h.flatten(1)  # (B, conv_channels * n_tiles)
        pn = self.to_pn(h)  # (B, n_pn)          -- Linear -> PN 層
        kc = F.relu(self.pn_to_kc(pn))  # (B, n_kc)  -- MaskedLinear(PN -> KC) + ReLU
        mbon = self.kc_to_mbon(kc)  # (B, n_mbon)  -- MaskedLinear(KC -> MBON)
        logits = self.to_logits(mbon)  # (B, n_tiles) -- Linear -> 34 logits
        return logits, {"pn": pn, "kc": kc, "mbon": mbon}
