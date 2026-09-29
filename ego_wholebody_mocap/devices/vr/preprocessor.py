from __future__ import annotations

import numpy as np

from ego_wholebody_mocap.devices.vr.constants import grd_yup2grd_zup


def mat_update(prev_mat, mat):
    if np.linalg.det(mat) == 0:
        return prev_mat
    return mat


def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def _transform_matrix_stack(matrices, basis, basis_inv):
    matrices = np.asarray(matrices, dtype=np.float64).reshape(-1, 4, 4)
    return np.matmul(np.matmul(basis, matrices), basis_inv)


class VuerPreprocessor:
    def __init__(self):
        self.vuer_head_mat = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 1.5],
                [0, 0, 1, -0.2],
                [0, 0, 0, 1],
            ]
        )
        self.body_matrices = {}

    def process(self, vr_body_streamer):
        self.vuer_head_mat = vr_body_streamer.head_matrix.copy()

        basis_inv = fast_mat_inv(grd_yup2grd_zup)
        self.body_matrices = {
            name: grd_yup2grd_zup @ mat @ basis_inv
            for name, mat in vr_body_streamer.body_matrices.items()
        }

        head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ basis_inv

        left_hand_matrices_world = _transform_matrix_stack(
            vr_body_streamer.left_hand_matrices.copy(),
            grd_yup2grd_zup,
            basis_inv,
        )
        right_hand_matrices_world = _transform_matrix_stack(
            vr_body_streamer.right_hand_matrices.copy(),
            grd_yup2grd_zup,
            basis_inv,
        )

        left_key_points_world = left_hand_matrices_world[:, :3, 3].copy()
        right_key_points_world = right_hand_matrices_world[:, :3, 3].copy()

        return (
            head_mat,
            self.body_matrices,
            left_key_points_world,
            right_key_points_world,
            left_hand_matrices_world,
            right_hand_matrices_world,
        )

    def get_hand_gesture(self, tv):
        left_key_points_world = np.concatenate(
            [
                tv.left_hand_key_points_world.copy().T,
                np.ones((1, tv.left_hand_key_points_world.shape[0])),
            ]
        )
        right_key_points_world = np.concatenate(
            [
                tv.right_hand_key_points_world.copy().T,
                np.ones((1, tv.right_hand_key_points_world.shape[0])),
            ]
        )

        left_key_points_world = (grd_yup2grd_zup @ left_key_points_world)[0:3, :].T
        right_key_points_world = (grd_yup2grd_zup @ right_key_points_world)[0:3, :].T
        all_fingers = np.concatenate([left_key_points_world, right_key_points_world], axis=0)

        return all_fingers
