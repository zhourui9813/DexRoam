from __future__ import annotations

from enum import Enum, auto
from typing import Dict

import numpy as np


class Pinch(Enum):
    """Common pinch combinations; can be extended as needed."""

    THUMB_INDEX = auto()
    THUMB_MIDDLE = auto()
    THUMB_RING = auto()
    THUMB_PINKY = auto()


class GestureRecognizer:
    """
    Gesture recognizer for 21-point hand model.

    Parameters
    ----------
    finger_tip_indices : Dict[str, int]
        Mapping from finger names to fingertip indices in the 21-point model.
    """

    def __init__(
        self,
        finger_tip_indices: Dict[str, int] = None,
        pos_coeff: float = 1.2,
        neg_coeff: float = 0.8,
    ):
        default_indices = {
            "thumb": 4,
            "index": 9,
            "middle": 14,
            "ring": 19,
            "pinky": 24,
        }
        self.tip_idx = finger_tip_indices or default_indices
        self.pos_coeff = pos_coeff
        self.neg_coeff = neg_coeff

    def detect(self, hand_mat: np.ndarray) -> Dict[Pinch, bool]:
        ref_len = np.linalg.norm(hand_mat[1] - hand_mat[2])
        pos_th = ref_len / self.pos_coeff
        neg_th = ref_len / self.neg_coeff

        def _close(a, b, th):
            idx_a, idx_b = self.tip_idx[a], self.tip_idx[b]
            return np.linalg.norm(hand_mat[idx_a] - hand_mat[idx_b]) < th

        results = {}
        for finger in ("index", "middle", "ring", "pinky"):
            pinch = getattr(Pinch, f"THUMB_{finger.upper()}")
            if not _close("thumb", finger, pos_th):
                results[pinch] = False
                continue

            other_fingers = [
                name for name in self.tip_idx
                if name not in ("thumb", finger)
            ]

            thumb_far_from_all_others = True
            for other in other_fingers:
                if _close("thumb", other, neg_th):
                    thumb_far_from_all_others = False
                    break

            results[pinch] = thumb_far_from_all_others
        return results

    def is_pinch(self, hand_mat: np.ndarray, pinch: Pinch) -> bool:
        return self.detect(hand_mat)[pinch]
