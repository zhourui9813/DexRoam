from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from ego_wholebody_mocap.gestures.capture_state_machine import (
    CaptureAction,
    CaptureState,
    CaptureUpdate,
    GestureCaptureController,
)
from ego_wholebody_mocap.config import (
    ABORT_DISCARD,
    ABORT_OVERFLOW,
    BACKLOG_BUDGET_GIB,
    CAMERA_CAPTURE_FPS,
    OUTPUT_DIR,
    STREAM_MODE,
    VR_IMAGE_HEIGHT,
    VR_IMAGE_POSITION,
    VR_PUSH_FPS,
    ZED_RES,
    ZED_STORAGE_MODE,
)
from ego_wholebody_mocap.devices.zed import (
    ZedCameraSession,
    apply_zed_stream_frame,
    require_zed,
)
from ego_wholebody_mocap.paths import (
    PROJECT_ROOT,
    _episode_path,
)
from ego_wholebody_mocap.storage.episode_manager import EpisodeCaptureManager
from ego_wholebody_mocap.storage.hdf5_writer import require_h5py
from ego_wholebody_mocap.ui.hud import (
    _display_status,
    _status_name,
    draw_status_on_tele,
)
from ego_wholebody_mocap.ui.state import (
    RuntimeDisplayState,
    _set_status_feedback,
    _update_loop_fps,
)


def _resolve_configured_output_dir(output_dir: str | Path) -> Path:
    base_output_dir = Path(output_dir).expanduser()
    if not base_output_dir.is_absolute():
        base_output_dir = PROJECT_ROOT / base_output_dir

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = base_output_dir / f"vuer_teleop_jpeg_{stamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    return run_output_dir.resolve()


def _vr_push_sleep_s_from_fps(vr_push_fps: float) -> float:
    vr_push_fps = max(float(vr_push_fps), 0.0)
    if vr_push_fps <= 0.0:
        return 0.0
    return 1.0 / vr_push_fps


def _print_startup_config(
    output_dir: Path,
    output_prefix: str | None,
    *,
    vr_image_height: float,
    vr_image_position: tuple[float, float, float],
) -> None:
    example_output_path = _episode_path(output_dir, output_prefix, 1, ".h5")
    vr_push_fps = max(float(VR_PUSH_FPS), 0.0)
    vr_push_sleep_s = _vr_push_sleep_s_from_fps(vr_push_fps)
    print(f"Confirmed episodes will be written under {output_dir}", flush=True)
    print(f"Example output path: {example_output_path}", flush=True)
    print(
        "Gestures: left THUMB_INDEX=start after delay, left THUMB_MIDDLE=stop/cancel, "
        "right THUMB_RING=save, right THUMB_MIDDLE=discard.",
        flush=True,
    )
    print(
        f"ZED config: res={ZED_RES}, "
        f"fps={CAMERA_CAPTURE_FPS}, stream_mode={STREAM_MODE}",
        flush=True,
    )
    print(f"VR push FPS: {vr_push_fps:.1f} (sleep={vr_push_sleep_s:.4f}s)", flush=True)
    print(
        f"VR image plane: height={float(vr_image_height):.3f}, position={tuple(vr_image_position)}",
        flush=True,
    )
    print(f"ZED writer mode: {ZED_STORAGE_MODE}", flush=True)
    print(f"Backlog budget: {BACKLOG_BUDGET_GIB:.2f} GiB", flush=True)
    print("Press Ctrl+C to stop.", flush=True)


def _reset_capture_to_idle(
    capture_controller: GestureCaptureController,
    display_state: RuntimeDisplayState,
) -> None:
    capture_controller.set_state(CaptureState.IDLE)
    display_state.last_status_name = None


def _handle_capture_action(
    *,
    gesture_update: CaptureUpdate,
    capture_manager: EpisodeCaptureManager,
    capture_controller: GestureCaptureController,
    display_state: RuntimeDisplayState,
    loop_monotonic: float,
    output_dir: Path,
    output_prefix: Optional[str],
    frame_shape: tuple[int, int, int],
    body_joint_names: tuple[str, ...],
    zed_res: str,
    fps: int,
    stream_mode: str,
    zed_storage_mode: str,
) -> None:
    status_name = _status_name(
        gesture_update.state,
        has_pending_episode=capture_manager.has_pending_episode(),
    )
    if status_name != display_state.last_status_name:
        if status_name == "Arming":
            print(f"[state] {gesture_update.status_text}", flush=True)
        else:
            print(f"[state] {status_name}", flush=True)
        display_state.last_status_name = status_name

    if gesture_update.action == CaptureAction.CANCEL_START:
        print("[record] Start cancelled.", flush=True)
        _set_status_feedback(display_state, "Start cancelled", loop_monotonic)

    elif gesture_update.action == CaptureAction.START_RECORDING:
        episode_idx = capture_manager.start_recording(
            frame_shape=frame_shape,
            body_joint_names=body_joint_names,
            zed_res=zed_res,
            fps=fps,
            stream_mode=stream_mode,
            zed_storage_mode=zed_storage_mode,
        )
        if episode_idx is not None:
            print(
                f"[record] Recording episode {episode_idx} -> "
                f"{_episode_path(output_dir, output_prefix, episode_idx, '.pending.h5')}",
                flush=True,
            )

    elif gesture_update.action == CaptureAction.STOP_RECORDING:
        episode_idx, result = capture_manager.stop_recording()
        if episode_idx is not None:
            if result is not None and result.capture_status == "stopped":
                print(
                    f"[record] Episode {episode_idx} captured "
                    f"head={result.head_count} body={result.body_count} zed={result.zed_count}. "
                    "AwaitDecision: right ring=save / right middle=discard",
                    flush=True,
                )
            else:
                error_path = _episode_path(output_dir, output_prefix, episode_idx, ".error.h5")
                print(
                    f"[error] Episode {episode_idx} failed during stop -> {error_path}",
                    flush=True,
                )
                _set_status_feedback(display_state, "Writer error", loop_monotonic)
                _reset_capture_to_idle(capture_controller, display_state)

    elif gesture_update.action == CaptureAction.SAVE_EPISODE:
        episode_idx, output_path, result = capture_manager.save_pending()
        if episode_idx is not None and output_path is not None and result is not None:
            print("[state] Saved", flush=True)
            print(
                f"[record] Saved episode {episode_idx} "
                f"(head={result.head_count}, body={result.body_count}, zed={result.zed_count}) -> {output_path}",
                flush=True,
            )
            _set_status_feedback(display_state, f"Saved episode{episode_idx:04d}", loop_monotonic)
            _reset_capture_to_idle(capture_controller, display_state)

    elif gesture_update.action == CaptureAction.DISCARD_EPISODE:
        episode_idx, pending_path = capture_manager.discard_pending()
        if episode_idx is not None:
            print("[state] Discarded", flush=True)
            print(
                f"[record] Discarded episode {episode_idx} ({pending_path}).",
                flush=True,
            )
            _set_status_feedback(display_state, "Discarded episode", loop_monotonic)
            _reset_capture_to_idle(capture_controller, display_state)


def _submit_pose_sample(
    *,
    capture_manager: EpisodeCaptureManager,
    teleop,
    head_mat: np.ndarray,
    body_matrices: dict[str, np.ndarray],
    left_hand_key_points_world: np.ndarray,
    right_hand_key_points_world: np.ndarray,
    left_hand_matrices_world: np.ndarray,
    right_hand_matrices_world: np.ndarray,
) -> None:
    active = capture_manager.active
    if active is None:
        return

    sample_monotonic_ns = time.monotonic_ns()
    sample_wall_ns = time.time_ns()
    active.broker.submit_head(
        sample_monotonic_ns=sample_monotonic_ns,
        sample_wall_ns=sample_wall_ns,
        head_event_monotonic_ns=teleop.vr_streamer.head_event_monotonic_ns,
        head_event_wall_ns=teleop.vr_streamer.head_event_wall_ns,
        head_mat=head_mat,
    )
    active.broker.submit_body(
        sample_monotonic_ns=sample_monotonic_ns,
        sample_wall_ns=sample_wall_ns,
        body_event_monotonic_ns=teleop.vr_streamer.body_event_monotonic_ns,
        body_event_wall_ns=teleop.vr_streamer.body_event_wall_ns,
        body_matrices=body_matrices,
        left_hand_key_points_world=left_hand_key_points_world,
        right_hand_key_points_world=right_hand_key_points_world,
        left_hand_matrices_world=left_hand_matrices_world,
        right_hand_matrices_world=right_hand_matrices_world,
    )


def _capture_zed_frame(
    *,
    zed_session: ZedCameraSession,
    composed_frame: np.ndarray,
    fallback_frame: np.ndarray,
    per_eye_width: int,
    stream_mode: str,
    capture_manager: EpisodeCaptureManager,
    capture_controller: GestureCaptureController,
    display_state: RuntimeDisplayState,
    loop_monotonic: float,
) -> None:
    sbs = zed_session.grab_side_by_side()
    if sbs is None:
        composed_frame[:] = fallback_frame
        return

    apply_zed_stream_frame(sbs, composed_frame, per_eye_width, stream_mode)

    active = capture_manager.active
    if active is None:
        return

    zed_frame_rgb = cv2.cvtColor(sbs, cv2.COLOR_BGRA2RGB)
    if active.broker.submit_zed_frame(
        frame_rgb_u8=zed_frame_rgb,
        capture_monotonic_ns=time.monotonic_ns(),
        capture_wall_ns=time.time_ns(),
        image_ts_ns=zed_session.image_timestamp_ns(),
    ):
        return

    episode_idx, result, error_path = capture_manager.abort_active(ABORT_OVERFLOW)
    if episode_idx is not None:
        print(
            f"[error] Episode {episode_idx} overflowed backlog budget -> {error_path}",
            flush=True,
        )
        if result is not None:
            print(
                f"[error] Counts before abort: "
                f"head={result.head_count} body={result.body_count} zed={result.zed_count}",
                flush=True,
            )
    _set_status_feedback(display_state, "Overflow error", loop_monotonic)
    _reset_capture_to_idle(capture_controller, display_state)


def _handle_writer_failure(
    *,
    capture_manager: EpisodeCaptureManager,
    capture_controller: GestureCaptureController,
    display_state: RuntimeDisplayState,
    loop_monotonic: float,
) -> None:
    failure = capture_manager.poll_active_failure()
    if failure is None:
        return

    episode_idx, result, error_path = capture_manager.abort_active(ABORT_DISCARD)
    if episode_idx is not None:
        print(f"[error] Writer failed for episode {episode_idx} -> {error_path}", flush=True)
        if result is not None and result.error_message:
            print(f"[error] Writer detail: {result.error_message}", flush=True)
    _set_status_feedback(display_state, "Writer error", loop_monotonic)
    _reset_capture_to_idle(capture_controller, display_state)


def main() -> None:
    output_dir = _resolve_configured_output_dir(OUTPUT_DIR)
    output_prefix = None
    vr_push_sleep_s = _vr_push_sleep_s_from_fps(VR_PUSH_FPS)
    vr_image_height = float(VR_IMAGE_HEIGHT)
    vr_image_position = tuple(float(value) for value in VR_IMAGE_POSITION)
    _print_startup_config(
        output_dir,
        output_prefix,
        vr_image_height=vr_image_height,
        vr_image_position=vr_image_position,
    )

    require_zed()
    require_h5py()

    zed_session: Optional[ZedCameraSession] = None
    teleop = None

    from ego_wholebody_mocap.devices.vr.vuer_recorder import VuerRecorder

    capture_manager = EpisodeCaptureManager(
        output_dir=output_dir,
        output_prefix=output_prefix,
        backlog_budget_bytes=int(BACKLOG_BUDGET_GIB * (1024 ** 3)),
        zed_storage_mode=ZED_STORAGE_MODE,
    )
    capture_controller = GestureCaptureController()

    display_state = RuntimeDisplayState()

    try:
        zed_session = ZedCameraSession(
            zed_res=ZED_RES,
            fps=CAMERA_CAPTURE_FPS,
        )
        image_height = zed_session.image_height
        per_eye_width = zed_session.per_eye_width
        image_aspect = per_eye_width / image_height

        teleop = VuerRecorder(
            resolution=(image_height, per_eye_width),
            image_push_sleep_s=vr_push_sleep_s,
            image_aspect=image_aspect,
            vr_image_height=vr_image_height,
            vr_image_position=vr_image_position,
        )
        teleop.img_array[:] = 0
        composed_frame = np.empty_like(teleop.img_array)

        while True:
            (
                head_mat,
                body_matrices,
                left_hand_key_points_world,
                right_hand_key_points_world,
                left_hand_matrices_world,
                right_hand_matrices_world,
            ) = teleop.step()

            loop_monotonic = time.monotonic()
            _update_loop_fps(display_state, loop_monotonic)

            gesture_update = capture_controller.update(
                loop_monotonic,
                left_hand_key_points_world,
                right_hand_key_points_world,
            )
            _handle_capture_action(
                gesture_update=gesture_update,
                capture_manager=capture_manager,
                capture_controller=capture_controller,
                display_state=display_state,
                loop_monotonic=loop_monotonic,
                output_dir=output_dir,
                output_prefix=output_prefix,
                frame_shape=zed_session.frame_shape,
                body_joint_names=tuple(teleop.vr_streamer.body_joint_names),
                zed_res=ZED_RES,
                fps=CAMERA_CAPTURE_FPS,
                stream_mode=STREAM_MODE,
                zed_storage_mode=ZED_STORAGE_MODE,
            ) #state machine monitor
            _submit_pose_sample(
                capture_manager=capture_manager,
                teleop=teleop,
                head_mat=head_mat,
                body_matrices=body_matrices,
                left_hand_key_points_world=left_hand_key_points_world,
                right_hand_key_points_world=right_hand_key_points_world,
                left_hand_matrices_world=left_hand_matrices_world,
                right_hand_matrices_world=right_hand_matrices_world,
            ) # pub the collected data to save
            _capture_zed_frame(
                zed_session=zed_session,
                composed_frame=composed_frame,
                fallback_frame=teleop.img_array,
                per_eye_width=per_eye_width,
                stream_mode=STREAM_MODE,
                capture_manager=capture_manager,
                capture_controller=capture_controller,
                display_state=display_state,
                loop_monotonic=loop_monotonic,
            ) # pub the zed to save
            _handle_writer_failure(
                capture_manager=capture_manager,
                capture_controller=capture_controller,
                display_state=display_state,
                loop_monotonic=loop_monotonic,
            )

            primary_status = _display_status(
                gesture_update.state,
                gesture_update.status_text,
                has_pending_episode=capture_manager.has_pending_episode(),
                feedback_text=display_state.feedback_text,
                now_monotonic=loop_monotonic,
                feedback_deadline=display_state.feedback_deadline,
            ) 
            draw_status_on_tele(
                composed_frame,
                per_eye_width,
                status_text=primary_status,
                episode_text=capture_manager.display_episode_text(),
                duration_text=capture_manager.display_episode_duration_text(loop_monotonic),
                loop_fps_text=f"{display_state.loop_fps:.1f}" if display_state.loop_fps > 0.0 else "--.-",
                push_fps_text=f"{teleop.vr_streamer.push_fps:.1f}" if teleop.vr_streamer.push_fps > 0.0 else "--.-",
            ) ###display the task info in the vr
            np.copyto(teleop.img_array, composed_frame)

    except KeyboardInterrupt:
        print("\nExit.", flush=True)
    finally:
        for message in capture_manager.discard_unsaved():
            print(message, flush=True)

        if teleop is not None:
            teleop.close()
        if zed_session is not None:
            zed_session.close()


if __name__ == "__main__":
    main()
