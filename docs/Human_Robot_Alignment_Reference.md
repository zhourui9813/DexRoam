# Human-Robot Alignment Reference

## 1. Overview

Raw human demonstrations cannot be used directly for robot policy training because humans and robots differ in morphology, control interfaces, and execution timescales. DexRoam resolves these mismatches through three explicit alignment stages: embodiment, action-semantic, and temporal alignment. This package implements embodiment alignment and temporal alignment; action-semantic alignment is applied during downstream data conversion. See the [top-level README](../README.md#-human-robot-alignment) for installation and quick-start commands.

## 2. Usage

| Argument | Description |
| --- | --- |
| `INPUT` | Input HDF5 file or directory |
| `-o PATH`, `--output PATH` | Output file or directory. Defaults to the input name with `_astribot_retargeted` appended |
| `--recursive` | Search input directories and `--robot-dir` recursively |
| `--overwrite` | Replace existing outputs instead of skipping or raising an error |
| `--config-dir PATH` | Use a directory containing custom `xhand_left.yml` and `xhand_right.yml` files |
| `--no-write-images` | Omit all `images_dict` datasets |
| `--crop HEIGHTxWIDTH` | Center-crop each stereo eye independently before writing images |
| `--resize HEIGHTxWIDTH` | Resize each stereo eye independently; can be combined with `--crop` |

## 3. Temporal Resampling

Resampling is selected by one of three mutually exclusive options. If none is given, trajectories keep their original length.

| Mode | Option | Output length |
| --- | --- | --- |
| Robot average length (default in the paper) | `--robot-dir PATH` | Rounded mean length of the robot episodes |
| Playback-speed factor | `--speed-factor FLOAT` | Resampled relative to each episode's own length; the factor must be finite and greater than 0 |
| Fixed target length | `--target-length N` | Exactly `N` frames; `N` must be at least 2 |

`--robot-dir` reads only episode lengths from the robot files; robot poses are never used for alignment. When robot demonstrations are unavailable, use `--speed-factor` or `--target-length` instead.

### 3.1 Examples

```bash
# Robot average length (default in the paper)
python -m dexroam_alignment path/to/human_dir -o path/to/output_dir --recursive \
  --robot-dir path/to/robot_dir

# Playback-speed factor
python -m dexroam_alignment path/to/human_dir -o path/to/output_dir --recursive \
  --speed-factor 2.0

# Fixed target length
python -m dexroam_alignment path/to/human_dir -o path/to/output_dir --recursive \
  --target-length 120
```

### 3.2 Interpolation Rules

| Data | Representation | Resampling method |
| --- | --- | --- |
| Pose7 datasets | 3D position + quaternion | Position: linear interpolation. Orientation: sign-align consecutive quaternions (`q` and `-q` encode the same rotation), then SLERP |
| Mobile-base pose | Planar SE(2) pose `(x, y, yaw)` | `x`, `y`: linear interpolation. `yaw`: unwrapped before linear interpolation, so headings near ±π follow the short arc instead of jumping |
| Other floating-point arrays | — | Linear interpolation |
| Images | — | Nearest-neighbor frame selection |