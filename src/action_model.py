"""Phase 2 Part 4: a multi-head architecture covering discard/riichi/tsumo/
kan/kyuushu (SELF decisions) and pass/pon/chii/openkan/ron (DISCARD_REACTION
decisions).

  mahjong feature (34 x 34)
    -> Conv1d front-end
    -> Linear -> PN layer
    -> MaskedLinear(PN -> KC)
    -> ReLU
    -> MaskedLinear(KC -> MBON)
    -> three heads:
         discard_head  (34)                          which tile (shared by DISCARD/RIICHI)
         self_type_head (len(SELF_ACTION_TYPES))      which action for a SELF decision
         reaction_head  (len(REACTION_ACTION_TYPES))  which action for a REACTION decision

The two hidden layers (PN->KC->MBON) keep the connectome mask — that's
still the whole point of the project. The heads added on top just decide
"which output space to use," without touching the mask itself.
"""

import torch
import torch.nn.functional as F
from torch import nn

from model import MaskedLinear

SELF_ACTION_TYPES = ["DISCARD", "RIICHI", "TSUMO", "KAN", "KYUUSHU"]
REACTION_ACTION_TYPES = ["PASS", "PON", "CHII", "OPEN_KAN", "RON"]


class ActionNet(nn.Module):
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

        self.discard_head = nn.Linear(n_mbon, n_tiles)
        self.self_type_head = nn.Linear(n_mbon, len(SELF_ACTION_TYPES))
        self.reaction_head = nn.Linear(n_mbon, len(REACTION_ACTION_TYPES))

    def trunk_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.conv(x)
        h = h.flatten(1)
        pn = self.to_pn(h)
        kc = F.relu(self.pn_to_kc(pn))
        mbon = self.kc_to_mbon(kc)
        return mbon, {"pn": pn, "kc": kc, "mbon": mbon}

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk_with_activations(x)[0]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mbon = self.trunk(x)
        return {
            "discard": self.discard_head(mbon),
            "self_type": self.self_type_head(mbon),
            "reaction": self.reaction_head(mbon),
        }

    def forward_with_activations(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        mbon, activations = self.trunk_with_activations(x)
        outputs = {
            "discard": self.discard_head(mbon),
            "self_type": self.self_type_head(mbon),
            "reaction": self.reaction_head(mbon),
        }
        return outputs, activations
