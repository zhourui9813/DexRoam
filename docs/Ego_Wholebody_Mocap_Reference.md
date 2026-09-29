# Ego Whole-Body Mocap System Reference

This document is the technical reference for `ego_wholebody_mocap`: code structure, Quest 3 and ZED Mini data flow, capture and storage configuration, the headset status display, and common runtime issues. For installation, hardware setup, and demonstration collection, see the [top-level README](../README.md#-human-motion-capture).

## 1. Code Structure

```text
ego_wholebody_mocap/
├── __main__.py             # `python -m ego_wholebody_mocap` entry point
├── main.py                 # Main acquisition loop and subsystem coordination
├── config.py               # User-facing capture, streaming, and storage parameters
├── devices/                # ZED camera and Quest/Vuer interfaces
├── gestures/               # Gesture recognition and recording state control
├── storage/                # Episode management and asynchronous HDF5 writing
└── ui/                     # Headset status overlay and runtime display state
```

## 2. System Architecture and Data Flow

```text
┌──────────┐    poses     ┌──────────┐    poses     ┌──────────────┐
│ Quest 3  │ ───────────► │   Vuer   │ ───────────► │              │
│          │ ◄─────────── │ process  │ ◄─────────── │              │
└──────────┘   preview    └──────────┘   preview    │     Main     │
                                                    │   process    │
┌──────────┐            stereo frames               │              │
│ ZED Mini │ ─────────────────────────────────────► │              │
└──────────┘                                        └──────┬───────┘
                                                           │ poses + frames
                                                           ▼
                                                    ┌──────────────┐
                                                    │    Writer    │ ───► HDF5 file
                                                    │   process    │
                                                    └──────────────┘
```

The system runs three processes:

- **Main process**: the central coordinator. It reads the latest head, body, and hand poses from Quest 3, converts them to the dataset coordinate convention, and collects stereo frames from the ZED Mini. It also recognizes pinch gestures to start, stop, save, or discard an episode, and sends the camera preview back to the headset through Vuer.
- **Vuer process**: hosts the WebXR session with Quest 3. Running it separately keeps headset communication from delaying the acquisition loop.
- **Writer process**: during recording, receives tracking samples and raw frames (frames are buffered in shared memory, see [3.3](#33-asynchronous-writer-and-backlog)), encodes the images, and writes the HDF5 file incrementally so disk I/O never interrupts capture.

## 3. `config.py` Parameters

All capture, streaming, and storage settings are defined in `ego_wholebody_mocap/config.py`.

### 3.1 Output

| Parameter | Default | Description |
| --- | --- | --- |
| `OUTPUT_DIR` | `"data/test_data"` | Base output directory; relative paths resolve from the repository root. Each launch creates a timestamped `vuer_teleop_jpeg_YYYYMMDD_HHMMSS/` subdirectory |

### 3.2 ZED Capture, Streaming, and Headset Display

| Parameter | Default | Description |
| --- | --- | --- |
| `ZED_RES` | `"HD1080"` | `"VGA"`, `"HD720"`, `"HD1080"`, or `"HD2K"`. Higher resolutions add image detail but increase shared-memory use, encoder load, and disk bandwidth |
| `CAMERA_CAPTURE_FPS` | `30` | Requested ZED frame rate; must be supported by the selected resolution |
| `STREAM_MODE` | `"left"` | `"left"`, `"right"`, or `"both"`. Single-eye modes show the same image to both eyes; `"both"` preserves stereo viewing. Stored frames are always full side-by-side stereo |
| `VR_PUSH_FPS` | `20.0` | Target rate for pushing preview images to the headset. `0` disables throttling and pushes as fast as the Vuer loop allows. Lower values reduce network and encoding load but make the view less smooth |
| `VR_IMAGE_HEIGHT` | `4.0` | Height of each virtual image plane in Vuer world units. Increase for a larger field of view |
| `VR_IMAGE_POSITION` | `(0.0, -1.0, 3)` | Image-plane position `(x, y, z)` in Vuer world coordinates. Adjust carefully so the image sits comfortably in front of the user |

### 3.3 Asynchronous Writer and Backlog

| Parameter | Default | Description |
| --- | --- | --- |
| `BACKLOG_BUDGET_GIB` | `8.0` | Memory budget for raw ZED frames not yet written. If no additional slot fits within it, the active episode is aborted to an `.error.h5` file |
| `FRAME_SLOTS_PER_SLAB` | `32` | Frames per shared-memory slab. Larger slabs mean fewer allocations but more memory committed per growth step |
| `INITIAL_FRAME_SLABS` | `2` | Slabs allocated when recording starts (64 frame slots with the defaults) |

### 3.4 HDF5 Image Encoding

| Parameter | Default | Description |
| --- | --- | --- |
| `ZED_STORAGE_MODE` | `"jpeg"` | `"jpeg"` stores variable-length compressed bytes; `"raw_uint8"` stores full RGB arrays |
| `JPEG_QUALITY` | `85` | OpenCV JPEG quality (`0`–`100`), used only in `jpeg` mode. Higher values increase image quality, file size, and encoder/write load |

## 4. Runtime Status and Troubleshooting

### 4.1 Headset HUD

| Field | Description |
| --- | --- |
| `Status` | `Ready`, `Arming`, `Recording`, `AwaitDecision`, `Saving`, or a transient result/error message |
| `Episode` | Active, pending, or next episode number |
| `Duration` | Duration of the active or pending episode |
| `Loop FPS` | Smoothed main acquisition-loop rate |
| `Push FPS` | Smoothed Vuer image-push rate |

### 4.2 Common Issues

| Symptom | Likely cause | Action |
| --- | --- | --- |
| ZED fails to open | Temporary camera connection failure | Unplug and reconnect the ZED camera, then restart the capture program |
| Gestures trigger unreliably | Ambiguous pinch or hand-tracking loss | Keep the non-pinching fingers extended and confirm Quest hand tracking is active before recording |
| Streaming latency grows during long sessions | Quest headset overheating | Stop collection and let the headset cool down, then restart the headset and capture program |