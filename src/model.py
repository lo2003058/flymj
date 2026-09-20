"""Network architecture for mahjong discard prediction (following the
diagram in data/doc/task.md):

  mahjong feature (C x 34)
    -> Conv1d front-end
    -> Linear -> PN layer
    -> MaskedLinear(PN -> KC)
    -> ReLU
    -> MaskedLinear(KC -> MBON)
    -> Linear -> 34 logits

The ReLU only sits between the two MaskedLinear layers, following
task.md's diagram literally (how many layers the Conv1d front-end uses,
and whether it has ReLU, is this module's own implementation detail — not
specified in the diagram, added following common practice).
"""

import torch
import torch.nn.functional as F
from torch import nn


class MaskedLinear(nn.Module):
    """An nn.Linear whose weight is always zero wherever the connectome
    mask has no edge.

    The mask is a buffer (not a parameter), so it's never updated by
    training or the optimizer.
    """

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super().__init__()
        if mask.shape != (out_features, in_features):
            raise ValueError(f"mask shape {tuple(mask.shape)} doesn't match (out_features, in_features)="
                              f"{(out_features, in_features)}")
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
            raise ValueError(f"mask_pn_kc's KC dimension {n_kc} doesn't match mask_kc_mbon's KC dimension {n_kc2}")

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
        """Same as forward, but also returns the PN/KC/MBON intermediate
        outputs, for the UI to plot "which neuron is active right now."""
        h = self.conv(x)  # (B, conv_channels, n_tiles)
        h = h.flatten(1)  # (B, conv_channels * n_tiles)
        pn = self.to_pn(h)  # (B, n_pn)          -- Linear -> PN layer
        kc = F.relu(self.pn_to_kc(pn))  # (B, n_kc)  -- MaskedLinear(PN -> KC) + ReLU
        mbon = self.kc_to_mbon(kc)  # (B, n_mbon)  -- MaskedLinear(KC -> MBON)
        logits = self.to_logits(mbon)  # (B, n_tiles) -- Linear -> 34 logits
        return logits, {"pn": pn, "kc": kc, "mbon": mbon}
