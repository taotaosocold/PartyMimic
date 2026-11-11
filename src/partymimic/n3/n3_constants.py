"""Noetix N3 V3: the 2026-11-03 training MJCF and actuator parameters used by GR00T.

The model is prepared in data/n3_29dof_mjlab by tools/data/n3_assets.py. All 29 hinges, including
the head, are controlled in MJCF order. Named N3_0905 NPZ coordinates are mapped to the V3 names;
this changes their column order, not their geometry. Reference velocities and bodies come from
PartyMimic's usual resampling and FK.
"""

from __future__ import annotations

import os
import re

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from partymimic.robot import MotionArrays, RobotSpec

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MJCF_PATH = os.path.join(REPO, "data", "n3_29dof_mjlab", "n3_29dof.xml")
MOTION_DIR = os.path.join(REPO, "data", "motions", "n3", "seed")

JOINT_NAMES = (
    "waist_yaw_joint",
    "l_arm_shoulder_pitch_joint", "l_arm_shoulder_roll_joint", "l_arm_shoulder_yaw_joint",
    "l_arm_elbow_pitch_joint", "l_arm_elbow_yaw_joint", "l_arm_hand_roll_joint", "l_arm_hand_pitch_joint",
    "r_arm_shoulder_pitch_joint", "r_arm_shoulder_roll_joint", "r_arm_shoulder_yaw_joint",
    "r_arm_elbow_pitch_joint", "r_arm_elbow_yaw_joint", "r_arm_hand_roll_joint", "r_arm_hand_pitch_joint",
    "head_pitch_joint", "head_yaw_joint",
    "l_leg_hip_pitch_joint", "l_leg_hip_roll_joint", "l_leg_hip_yaw_joint", "l_leg_knee_pitch_joint",
    "l_leg_ankle_pitch_joint", "l_leg_ankle_roll_joint",
    "r_leg_hip_pitch_joint", "r_leg_hip_roll_joint", "r_leg_hip_yaw_joint", "r_leg_knee_pitch_joint",
    "r_leg_ankle_pitch_joint", "r_leg_ankle_roll_joint",
)  # fmt: skip

TRACKED_BODIES = (
    "base_link", "l_leg_hip_roll_link", "l_leg_knee_pitch_link", "l_leg_ankle_roll_link",
    "r_leg_hip_roll_link", "r_leg_knee_pitch_link", "r_leg_ankle_roll_link", "waist_yaw_link",
    "l_arm_shoulder_roll_link", "l_arm_elbow_pitch_link", "l_arm_hand_pitch_link",
    "r_arm_shoulder_roll_link", "r_arm_elbow_pitch_link", "r_arm_hand_pitch_link", "head_yaw_link",
)  # fmt: skip
KEYPOINT_BODIES = (
    "base_link", "head_yaw_link", "l_arm_hand_pitch_link", "r_arm_hand_pitch_link",
    "l_leg_ankle_roll_link", "r_leg_ankle_roll_link",
)
FOOT_BODIES = ("l_leg_ankle_roll_link", "r_leg_ankle_roll_link")

# (joint regex, stiffness, damping, effort limit, armature), matching GR00T's N3_ACTUATOR_GROUPS.
ACTUATORS = (
    (".*_leg_(hip_(pitch|roll|yaw)|knee_pitch)_joint", 200.0, 15.0, 145.0, 0.06086),
    (".*_leg_ankle_pitch_joint", 80.0, 4.0, 50.0, 0.0029075),
    (".*_leg_ankle_roll_joint", 50.0, 2.0, 50.0, 0.0029075),
    ("waist_yaw_joint", 80.0, 3.0, 145.0, 0.06086),
    (".*_arm_(shoulder_(pitch|roll|yaw)|elbow_pitch)_joint", 50.0, 3.0, 50.0, 0.0029075),
    (".*_arm_(elbow_yaw|hand_(roll|pitch))_joint", 20.0, 1.0, 20.0, 0.0019182),
    ("head_(pitch|yaw)_joint", 20.0, 3.0, 20.0, 0.0019182),
)
JOINT_FRICTIONLOSS = 0.01
DEFAULT_JOINT_POS = {
    ".*_leg_hip_pitch_joint": -0.1495,
    ".*_leg_knee_pitch_joint": 0.3215,
    ".*_leg_ankle_pitch_joint": -0.1720,
    "l_arm_shoulder_roll_joint": 0.2,
    "r_arm_shoulder_roll_joint": -0.2,
}

# The current local SEED files have short deployment names, already in DFS order. Resolve by
# name so this loader also accepts explicitly named BFS or shuffled files without double-reordering.
JOINT_ALIASES = {
    f"{side}_{old}_joint": f"{side}_{new}_joint"
    for side in ("l", "r")
    for old, new in (
        ("arm_pitch", "arm_shoulder_pitch"), ("arm_roll", "arm_shoulder_roll"),
        ("arm_yaw", "arm_shoulder_yaw"), ("elbow_pitch", "arm_elbow_pitch"),
        ("hand_yaw", "arm_elbow_yaw"), ("hand_roll", "arm_hand_roll"), ("hand_pitch", "arm_hand_pitch"),
        ("hip_pitch", "leg_hip_pitch"), ("hip_roll", "leg_hip_roll"), ("hip_yaw", "leg_hip_yaw"),
        ("knee_pitch", "leg_knee_pitch"), ("ankle_pitch", "leg_ankle_pitch"), ("ankle_roll", "leg_ankle_roll"),
    )
}


def actuator_of(joint: str) -> tuple[float, float, float, float]:
    for pat, *vals in ACTUATORS:
        if re.fullmatch(pat, joint):
            return tuple(vals)
    raise KeyError(joint)


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    for act in list(spec.actuators):
        spec.delete(act)
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
            joint.frictionloss = JOINT_FRICTIONLOSS
    return spec


def get_robot_cfg() -> EntityCfg:
    acts = tuple(
        BuiltinPositionActuatorCfg(target_names_expr=(pat,), stiffness=kp, damping=kd, effort_limit=eff, armature=arm)
        for pat, kp, kd, eff, arm in ACTUATORS
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.724), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.0}),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=acts, soft_joint_pos_limit_factor=0.9),
    )


def load_n3_npz(path: str) -> MotionArrays:
    """Root pose and joint coordinates in V3 MJCF order; body/velocity targets are recomputed by MotionLib."""
    with np.load(path, allow_pickle=False) as d:
        joints = d["joint_pos"]
        if "joint_names" in d:
            names = [JOINT_ALIASES.get(n, n) for n in d["joint_names"].astype(str).tolist()]
            if len(names) != len(JOINT_NAMES) or set(names) != set(JOINT_NAMES):
                raise ValueError(f"{path}: joint_names must identify the 29 N3 joints")
            joints = joints[:, [names.index(n) for n in JOINT_NAMES]]
        elif "joint_order" not in d or str(d["joint_order"].item()) != "depth_first":
            raise ValueError(f"{path}: N3 NPZ needs joint_names or joint_order='depth_first' in MJCF order")
        root = d["body_names"].astype(str).tolist().index("base_link") if "body_names" in d else 0
        p, q, fps = d["body_pos_w"][:, root], d["body_quat_w"][:, root], float(d["fps"].reshape(-1)[0])
        if joints.shape != (len(p), len(JOINT_NAMES)) or q.shape != (len(p), 4) or len(p) < 3:
            raise ValueError(f"{path}: expected matching root and 29-joint arrays with at least three frames")
        if not np.isfinite(fps) or fps <= 0 or not all(np.isfinite(x).all() for x in (p, q, joints)):
            raise ValueError(f"{path}: non-finite motion data or invalid fps")
        return p, q, joints, fps


MIRROR_PERMUTATION = tuple(
    JOINT_NAMES.index("r_" + n[2:] if n.startswith("l_") else "l_" + n[2:] if n.startswith("r_") else n)
    for n in JOINT_NAMES
)

SPEC = RobotSpec(
    name="n3",
    mjcf_path=MJCF_PATH,
    joint_names=JOINT_NAMES,
    tracked_bodies=TRACKED_BODIES,
    default_joint_pos=DEFAULT_JOINT_POS,
    action_scale={n: 0.25 * actuator_of(n)[2] / actuator_of(n)[0] for n in JOINT_NAMES},
    com_body="waist_yaw_link",
    com_range=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)),
    robot_cfg=get_robot_cfg,
    motion_dir=MOTION_DIR,
    motion_glob="**/*.npz",
    load_motion=load_n3_npz,
    keypoint_bodies=KEYPOINT_BODIES,
    foot_bodies=FOOT_BODIES,
    foot_height=0.066,  # seven foot capsules: centres z=-0.056, radius=0.010
    contact_height=0.078,
    mirror_permutation=MIRROR_PERMUTATION,
)
