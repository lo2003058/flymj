"""Phase 2 Part 5: inference wrapper around the multi-head action model,
for the UI to use."""

from dataclasses import dataclass

import numpy as np
import torch

from action_features import encode
from action_model import REACTION_ACTION_TYPES, SELF_ACTION_TYPES, ActionNet


def load_action_model(checkpoint_path: str, masks_npz, device: torch.device) -> ActionNet:
    # weights_only=False: this checkpoint is only ever produced locally by
    # train_action_model.py, not an untrusted downloaded file.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mask_pn_kc = torch.tensor(masks_npz["mask_pn_kc_real"], dtype=torch.float32, device=device)
    mask_kc_mbon = torch.tensor(masks_npz["mask_kc_mbon_real"], dtype=torch.float32, device=device)
    model = ActionNet(ckpt["n_channels"], ckpt["n_tiles"], mask_pn_kc, mask_kc_mbon).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@dataclass
class ActionPredictionResult:
    self_type_ranked: list[tuple[str, float]]  # highest to lowest
    reaction_ranked: list[tuple[str, float]]
    discard_ranked_hand_tiles: list[tuple[int, float]]
    activations: dict[str, np.ndarray]


def predict_action(
    model: ActionNet,
    device: torch.device,
    seat: int,
    hand_counts: list[int],
    hand_red_counts: list[int],
    meld_counts: list[list[int]],
    discard_counts: list[list[int]],
    riichi: list[bool],
    dora_tiles: list[int],
    round_wind: int,
    seat_wind: int,
    trigger_tile: int = -1,
) -> ActionPredictionResult:
    x = encode(
        seat=seat,
        hand_counts=hand_counts,
        hand_red_counts=hand_red_counts,
        meld_counts=meld_counts,
        discard_counts=discard_counts,
        riichi=riichi,
        dora_tiles=dora_tiles,
        round_wind=round_wind,
        seat_wind=seat_wind,
        trigger_tile=trigger_tile,
    )
    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        outputs, acts = model.forward_with_activations(x_t)
        self_type_probs = torch.softmax(outputs["self_type"], dim=1).squeeze(0).cpu().numpy()
        reaction_probs = torch.softmax(outputs["reaction"], dim=1).squeeze(0).cpu().numpy()
        discard_probs = torch.softmax(outputs["discard"], dim=1).squeeze(0).cpu().numpy()

    self_type_ranked = sorted(
        zip(SELF_ACTION_TYPES, (float(p) for p in self_type_probs)), key=lambda kv: kv[1], reverse=True
    )
    reaction_ranked = sorted(
        zip(REACTION_ACTION_TYPES, (float(p) for p in reaction_probs)), key=lambda kv: kv[1], reverse=True
    )

    hand_tiles = [kind for kind, count in enumerate(hand_counts) if count > 0]
    discard_ranked = sorted(
        ((kind, float(discard_probs[kind])) for kind in hand_tiles), key=lambda kv: kv[1], reverse=True
    )

    activations = {name: act.squeeze(0).cpu().numpy() for name, act in acts.items()}
    return ActionPredictionResult(
        self_type_ranked=self_type_ranked,
        reaction_ranked=reaction_ranked,
        discard_ranked_hand_tiles=discard_ranked,
        activations=activations,
    )
