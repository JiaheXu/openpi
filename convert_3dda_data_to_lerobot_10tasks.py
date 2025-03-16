"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        # "prompt":{
        #     "dtype": "string",
        #     "shape": (1,),
        # }
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 256, 256),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    # if Path(LEROBOT_HOME / repo_id).exists():
        # shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=3,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.imdecode(data, 1))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    
    data = np.load(ep_path, allow_pickle = True)
    # print("data: ", len(data))
    
    imgs_per_cam = {}
    
    head_cam = []
    left_cam = []
    right_cam = []

    state = []
    action = []

    for point in data:

        left_state = point['left_pos'][0:7]
        right_state = point['right_pos'][0:7]
        state.append( np.concatenate( (left_state, right_state)) )

        left_action = point['left_controller_pos'][0:7]
        right_action = point['right_controller_pos'][0:7]
        action.append( np.concatenate( (left_action, right_action)) )

        head_cam.append( np.transpose(point['head_rgb'],(2,0,1) ) )
        left_cam.append( np.transpose(point['left_rgb'],(2,0,1) ) )        
        right_cam.append( np.transpose(point['right_rgb'],(2,0,1) ) )

    state = np.array(state)
    state = torch.from_numpy(state)
    action = np.array(action)
    action = torch.from_numpy(action)
    # print("state: ", state.shape)
    velocity = None
    effort = None

    head_cam = np.array(head_cam)
    left_cam = np.array(left_cam)
    right_cam = np.array(right_cam)

    imgs_per_cam[ 'cam_high' ] = head_cam
    imgs_per_cam[ 'cam_left_wrist' ] = left_cam    
    imgs_per_cam[ 'cam_right_wrist' ] = right_cam

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    files: list[Path],
    task: str,
    episodes: list[int] | None = None,
    task_idx: int = 0,
) -> LeRobotDataset:


    if episodes is None:
        episodes = range(len(files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                # "prompt": task,
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)

        dataset.save_episode(task=task) #,task_index=task_idx)

    return dataset


def port_mobaloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "pickup_plate",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = True,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    tasks = ['handover_block', 'insert_battery', 'insert_marker_into_cup', 'lift_ball', 'open_marker', 'pickup_plate', 'stack_blocks', 'stack_bowls', 'straighten_rope', 'ziploc']
    for task_idx, task in enumerate(tasks):

        raw_dir = Path( "/ws/data/raw_demo/" + task + "/traj/" )
        files = sorted(raw_dir.glob("*.npy"))


        dataset = populate_dataset(
            dataset,
            files,
            task=task,
            episodes=episodes,
            task_idx = task_idx
        )
    print("dataset.tasks: ", dataset.meta.tasks)
    dataset.consolidate()

    dataset.push_to_hub()

if __name__ == "__main__":
    tyro.cli(port_mobaloha)
