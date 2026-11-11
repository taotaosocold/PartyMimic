"""View a checkpoint controlling one motion live in the mjlab simulator.

PYTHONPATH=src .venv/bin/python tools/play.py --ckpt checkpoints/g1_mlp.pt \
    --motion data/motions/g1/kungfu/kf_ground_fist_199.npz
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--motion", required=True, type=Path, help="one .npz or robot-native motion file")
    parser.add_argument("--robot", default=None, help="defaults to the checkpoint's robot")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed; 0.5 is half speed")
    parser.add_argument("--once", action="store_true", help="exit after one rollout instead of looping")
    parser.add_argument("--headless", action="store_true", help="run without a window, for checking a rollout")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    for path in (args.ckpt, args.motion):
        if not path.is_file():
            parser.error(f"file not found: {path}")
    if not args.headless:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            parser.error("no desktop display found; run this command in a graphical desktop terminal")
        # A native window needs GLFW even if the shell was configured for offscreen rendering.
        os.environ["MUJOCO_GL"] = "glfw"

    import mujoco
    import torch

    import partymimic.networks as networks
    from partymimic.dr import XiLayout
    from partymimic.env import TrackingSim, make_sim
    from partymimic.kinematics import Kinematics
    from partymimic.mdp import MAX_STEP_REWARD, MDP
    from partymimic.motion import CONTROL_DT, MotionLib, load_npz
    from partymimic.networks import Policy, policy_kwargs
    from partymimic.robot import get_robot

    if not torch.cuda.is_available():
        parser.error("CUDA is required by the mjlab tracking simulator")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda:0"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ck["args"]
    robot_name = args.robot or config.get("robot", "g1mj")
    if config.get("robot") and robot_name != config["robot"]:
        parser.error(f"checkpoint was trained with --robot {config['robot']}; use that robot")
    robot = get_robot(robot_name)
    kin = Kinematics(robot, dev)
    loader = load_npz if args.motion.suffix.lower() == ".npz" else robot.load_motion
    motion = MotionLib([str(args.motion)], kin, dev, loader)
    motion.set_interface(config.get("interface", "joint"), robot.tracked_bodies)
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev),
              XiLayout(robot, dev).dim, robot.sensing)
    K, H = config["goal_frames"], config["hist"]
    widths = lambda key: tuple(int(x) for x in config.get(key, "2048,1024,512,256").split(","))
    networks.AMP_DTYPE = torch.bfloat16 if config.get("amp", "none") == "bf16" else None
    policy = Policy(mdp.obs_dim, motion.goal_dim(K), robot.num_joints, hist=H,
                    enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"),
                    **policy_kwargs(config, robot)).to(dev)
    policy.load_state_dict(ck["pi"])
    policy.eval()
    env = TrackingSim(make_sim(robot), 1, robot, motion, mdp, dev, eval_mode=True)
    ids = torch.arange(1, device=dev)
    controls = {"paused": False, "reset": False}

    def on_key(key):
        if key == 32:  # Space
            controls["paused"] = not controls["paused"]
        elif key in (ord("R"), ord("r")):
            controls["reset"] = True

    def reset():
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(ids, motion.clip_start.clone())
        controls["reset"] = False

    reset()
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    viewer = None
    viewer_threads = []

    def update_view():
        # Display the actual GPU simulation state; the viewer does not simulate the robot itself.
        data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
        data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
        mujoco.mj_forward(model, data)
        if viewer is not None:
            with viewer.lock():
                viewer.cam.lookat[:2] = data.qpos[:2]
            viewer.sync()

    try:
        if not args.headless:
            import mujoco.viewer

            update_view()
            existing_threads = set(threading.enumerate())
            viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)
            viewer_threads = [thread for thread in threading.enumerate() if thread not in existing_threads]
            with viewer.lock():
                viewer.cam.distance = 3.5
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -15
                viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.7]
            print("Live policy playback. Space: pause/resume; R: restart; close window: exit.", flush=True)
        print(f"Robot: {robot_name}; motion: {args.motion.name}; "
              f"duration: {(motion.num_frames - 1) * CONTROL_DT:.2f}s", flush=True)
        hist, reward_sum = None, 0.0
        with torch.inference_mode():
            while viewer is None or viewer.is_running():
                start = time.monotonic()
                if controls["reset"]:
                    reset()
                    hist, reward_sum = None, 0.0
                if not controls["paused"]:
                    obs = env.observe()
                    hist = obs.unsqueeze(1).repeat(1, H, 1) if hist is None else torch.cat(
                        (hist[:, 1:], obs.unsqueeze(1)), dim=1)
                    result = env.step(policy(hist, motion.goal(env.f, K)))
                    reward_sum += result["r"].item()
                    env.f = torch.minimum(env.f, motion.clip_end - 1)
                if viewer is not None:
                    update_view()
                if bool(env.done_eval.item()):
                    norm = reward_sum / ((motion.num_frames - 1) * MAX_STEP_REWARD)
                    print(f"norm_return={norm:.4f}, failed={bool(env.failed_eval.item())}", flush=True)
                    if args.once or args.headless:
                        break
                    # Hold the terminal pose briefly, while keeping the window responsive.
                    hold_until = time.monotonic() + 0.6
                    while viewer.is_running() and time.monotonic() < hold_until and not controls["reset"]:
                        viewer.sync()
                        time.sleep(0.01)
                    reset()
                    hist, reward_sum = None, 0.0
                if viewer is not None:
                    time.sleep(max(0.0, CONTROL_DT / args.speed - (time.monotonic() - start)))
    finally:
        if viewer is not None:
            viewer.close()
            # Let the GLFW thread release its graphics context before Python/CUDA shutdown.
            for thread in viewer_threads:
                thread.join(timeout=5)


if __name__ == "__main__":
    main()
