from pathlib import Path
import random
from typing import Union
import train_memory
import train_vae
import vae
import memory
import os
import config
import numpy as np
import cv2
import torch
import mdn

dev = config.DEVICE


def frames_to_video(frames: Union[torch.Tensor, np.ndarray],
                    filename: Union[str, Path]):

    if isinstance(frames, torch.Tensor):
        frames = (frames.detach().cpu().permute(0, 2, 3, 1) *
                  255).long().numpy().astype(np.uint8)

    out = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*"DIVX"), 20,
                          (64, 64))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  #type: ignore
        out.write(frame)
    out.release()


def create_comparison():
    """ Choose a random replay from the folder and create videos
    """
    # load a replay
    replay_dir = config.REPLAY_DIR
    replays = sorted(replay_dir.glob("*.npz"), key=os.path.basename)
    video_dir = config.VIDEO_DIR

    n_replay = random.randint(0, len(replays) - 1)
    replay = replays[n_replay]
    print(f"Replay n: {n_replay}")
    data = np.load(replay)
    actual_frames = data["obs"]
    actions = data["action"]

    # save video of actual game
    frames_to_video(actual_frames, video_dir / "actual_game.avi")

    # encode every frame with the vae
    vae_model = vae.load_vae()
    vae_model.eval()
    vae_model.to(dev)
    tensor_frames = [
        torch.from_numpy(frame).float().permute(2, 0, 1).to(dev) / 255
        for frame in actual_frames
    ]
    vae_input = torch.stack(tensor_frames, dim=0).to(dev)
    z, _, _ = vae_model.encode(vae_input)

    # decode every frame with the vae
    decoded = vae_model.decode(z)

    # save the video of the decoded frames
    frames_to_video(decoded, video_dir / "decoded_game.avi")

    # use memory to predict the course of the game given the first frames
    mem = memory.load_memory()
    mem.eval()
    mem.to(dev)

    n_actual_frames = 100
    actions = torch.tensor(actions).to(dev)
    memory_input = torch.cat((z[:n_actual_frames], actions[:n_actual_frames]),
                             dim=-1).unsqueeze(1).to(dev)

    predicted_frames = tensor_frames[:n_actual_frames]
    # add 20 black frames to signal the end of the jumpstart
    for i in range(20):
        predicted_frames.append(torch.zeros(3, 64, 64).to(dev))

    for timestep in range(n_actual_frames, len(actual_frames)):
        with torch.no_grad():
            pi, sigma, mu = mem(memory_input)
            pred_next_z = mdn.sample(pi[-1], sigma[-1], mu[-1])
            pred_next_frame = vae_model.decode(pred_next_z)

        predicted_frames.append(pred_next_frame.squeeze().detach())
        pred_next_z_action = torch.cat(
            (pred_next_z, actions[timestep].unsqueeze(0)), dim=-1)
        memory_input = torch.cat(
            (memory_input, pred_next_z_action.unsqueeze(0)), dim=0)

    # save the video of the predicted frames
    frames_to_video(torch.stack(predicted_frames, dim=0),
                    video_dir / "predicted_game.avi")
