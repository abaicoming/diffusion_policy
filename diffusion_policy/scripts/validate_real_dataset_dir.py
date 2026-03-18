if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import argparse
import glob
import os
import pathlib

import av
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer


def check_exists(path: pathlib.Path, label: str, errors):
    if not path.exists():
        errors.append(f"Missing {label}: {path}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate a raw real-robot dataset directory before training."
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset root")
    parser.add_argument(
        "--required-lowdim-keys",
        nargs="+",
        default=["action", "timestamp", "robot_eef_pose"],
        help="Lowdim arrays that must exist in replay_buffer.zarr/data",
    )
    parser.add_argument(
        "--optional-lowdim-keys",
        nargs="+",
        default=["robot_joint", "robot_joint_vel"],
        help="Lowdim arrays that are nice to have but not required",
    )
    parser.add_argument(
        "--required-cameras",
        nargs="+",
        default=["0", "1"],
        help="Camera indices that must exist for every episode",
    )
    args = parser.parse_args()

    dataset = pathlib.Path(os.path.expanduser(args.dataset)).resolve()
    replay_path = dataset / "replay_buffer.zarr"
    video_root = dataset / "videos"

    errors = []
    warnings = []

    check_exists(dataset, "dataset root", errors)
    check_exists(replay_path, "replay_buffer.zarr", errors)
    check_exists(video_root, "videos directory", errors)

    if errors:
        for line in errors:
            print(line)
        raise SystemExit(1)

    expected_meta = [
        replay_path / ".zgroup",
        replay_path / "data" / ".zgroup",
        replay_path / "meta" / ".zgroup",
        replay_path / "meta" / "episode_ends" / ".zarray",
    ]
    for path in expected_meta:
        if not path.exists():
            errors.append(f"Missing zarr metadata file: {path}")

    for key in args.required_lowdim_keys:
        key_dir = replay_path / "data" / key
        if not check_exists(key_dir, f"array dir for {key}", errors):
            continue
        zarray_path = key_dir / ".zarray"
        if not zarray_path.exists():
            errors.append(f"Missing .zarray for required key '{key}': {zarray_path}")
        chunk_files = [p for p in key_dir.iterdir() if p.name != ".zarray"]
        if len(chunk_files) == 0:
            errors.append(f"No chunks found for required key '{key}': {key_dir}")

    for key in args.optional_lowdim_keys:
        key_dir = replay_path / "data" / key
        if not key_dir.exists():
            warnings.append(f"Optional key missing: {key}")
            continue
        zarray_path = key_dir / ".zarray"
        if not zarray_path.exists():
            warnings.append(f"Optional key missing .zarray: {zarray_path}")
        chunk_files = [p for p in key_dir.iterdir() if p.name != ".zarray"]
        if len(chunk_files) == 0:
            warnings.append(f"Optional key has no chunks: {key_dir}")

    episode_dirs = sorted([p for p in video_root.iterdir() if p.is_dir()], key=lambda p: int(p.name))
    if len(episode_dirs) == 0:
        errors.append(f"No episode video directories found under {video_root}")

    video_info = []
    for ep_dir in episode_dirs:
        for cam_idx in args.required_cameras:
            video_path = ep_dir / f"{cam_idx}.mp4"
            if not video_path.exists():
                errors.append(f"Missing camera video: {video_path}")
                continue
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                video_info.append(
                    (
                        str(video_path.relative_to(dataset)),
                        stream.codec_context.width,
                        stream.codec_context.height,
                        float(stream.average_rate) if stream.average_rate else None,
                    )
                )

    print(f"Dataset: {dataset}")
    print(f"Episodes with video: {len(episode_dirs)}")
    print("Video streams:")
    for row in video_info:
        print(f"  {row[0]} -> {row[1]}x{row[2]} @ {row[3]}")

    try:
        root = zarr.open(str(replay_path), mode="r")
        print(f"zarr root keys: {list(root.keys())}")
        if "data" in root:
            print(f"zarr data keys: {list(root['data'].keys())}")
        if "meta" in root:
            print(f"zarr meta keys: {list(root['meta'].keys())}")
    except Exception as e:
        errors.append(f"Failed to open zarr root via zarr.open: {type(e).__name__}: {e}")

    try:
        replay_buffer = ReplayBuffer.create_from_path(str(replay_path), mode="r")
        print(f"ReplayBuffer keys: {list(replay_buffer.data.keys())}")
        print(f"ReplayBuffer episodes: {replay_buffer.n_episodes}")
        print(f"ReplayBuffer steps: {replay_buffer.n_steps}")
    except Exception as e:
        errors.append(
            f"ReplayBuffer.create_from_path failed: {type(e).__name__}: {e}"
        )

    if warnings:
        print("Warnings:")
        for line in warnings:
            print(f"  {line}")

    if errors:
        print("Errors:")
        for line in errors:
            print(f"  {line}")
        raise SystemExit(1)

    print("Validation passed.")


if __name__ == "__main__":
    main()
