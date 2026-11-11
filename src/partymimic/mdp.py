"""Analytic MDP heads shared by the simulator and the world model (SuperMimic paper Appendix A).

State tuple ``S = (p, R, v, w, q, qd)``: root position, rotation matrix, world linear/angular
velocity of the root link frame, joint positions and velocities. ``J`` is the number of joints.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kinematics import Kinematics
from .motion import CONTROL_DT, SPARSE5_ERR, MotionLib
from .rotations import mat_to_6d, mm3, mv3, quat_to_mat, mat_to_quat, rot_angle_sq, so3_log, yaw_mat
from .sensing import SensingCfg, obs_noise


def state_dim(J: int) -> int:
    return 3 + 4 + 3 + 3 + J + J


def obs_err_dim(interface: str = "joint") -> int:
    """Size of the decoder-only block at the end of the observation (--interface sparse5: MDP.kp_err)."""
    return SPARSE5_ERR if interface == "sparse5" else 0


def obs_dim(J: int, interface: str = "joint") -> int:
    return J + J + 3 + 3 + J + 6 + obs_err_dim(interface)


def obs_scale(J: int, interface: str = "joint") -> torch.Tensor:
    # normalization of the obs before the policy (qd and angular velocity are large)
    return torch.cat(
        (torch.ones(J), torch.full((J,), 0.1), torch.full((3,), 0.25), torch.ones(3), torch.ones(J), torch.ones(6),
         torch.ones(obs_err_dim(interface)))  # fmt: skip
    )


MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    J = (x.shape[-1] - 13) // 2
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13 : 13 + J], x[..., 13 + J :]


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(mm3(H.transpose(-1, -2), R)),
            mv3(Rt, v),
            mv3(Rt, w),
            q,
            qd,
        ),
        -1,
    )


def inv_dim(J: int) -> int:
    return 1 + 6 + 3 + 3 + J + J



class MDP:
    def __init__(self, kin: Kinematics, motion: MotionLib, default_q: torch.Tensor, xi_dim: int,
                 sensing: SensingCfg = SensingCfg(), cfg: RewardCfg = RewardCfg()):  # fmt: skip
        self.kin, self.motion, self.cfg = kin, motion, cfg
        self.default_q = default_q
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        dev = default_q.device
        J = default_q.shape[0]
        self.num_j = J
        self.interface = motion.interface  # MotionLib.set_interface before the MDP is built
        self.obs_dim, self.xi_dim = obs_dim(J, self.interface), xi_dim
        self.critic_extra = J + len(kin.tracked_idx) * 3 + 3 + 6 + self.xi_dim
        self.obs_noise = obs_noise(sensing, J, self.interface).to(dev)
        self.obs_scale = obs_scale(J, self.interface).to(dev)
        # InstantTrack contact reward (training only): tracked-body indices of the feet, set by the trainer
        self.feet: list[int] = []

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi, f) -> torch.Tensor:
        """Noise-free proprioceptive observation (+ reference-relative root orientation)."""
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        dq0 = xi[..., 1 : 1 + q.shape[-1]]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        rel = mat_to_6d(mm3(Rt, self.motion.root_R[f]))
        o = (q - self.default_q - dq0, qd, mv3(Rt, w), g_b, a_prev, rel)
        if self.interface == "sparse5":
            o += (self.kp_err(R, q - dq0, f),)
        return torch.cat(o, -1)

    def kp_err(self, R, q, f) -> torch.Tensor:
        """--interface sparse5 observation block: wrist / ankle position errors (4 x 3), their so3_log orientation errors
        (4 x 3) and the head orientation error (3) of the robot (forward kinematics of the joint positions q for the
        root attitude R) against reference frame f, each side relative to its own pelvis heading frame."""
        m, kp = self.motion, self.motion.kp[1:]  # head, wrists, ankles
        bp, bR = self.kin.pose(mm3(yaw_mat(R).transpose(-1, -2), R), q, kp)
        Ht = yaw_mat(m.root_R[f]).transpose(-1, -2).unsqueeze(-3)
        rp = mv3(Ht, m.kp_p[f][..., 1:, :] - m.root_p[f].unsqueeze(-2))
        rR = mm3(Ht, m.kp_R[f][..., 1:, :, :])
        e_R = so3_log(mm3(rR.transpose(-1, -2), bR))
        return torch.cat(((bp - rp)[..., 1:, :].flatten(-2), e_R[..., 1:, :].flatten(-2), e_R[..., 0, :]), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    # ---------------- reward / termination ----------------
    def reward(self, S, f, a, a_prev, terms: bool = False, w_contact: float = 0.0):
        """Per-step reward (already multiplied by dt) comparing S with reference frame f.

        ``w_contact`` > 0 adds the InstantTrack contact term: a smooth foot-contact indicator (link origin
        below the robot's contact height) matched to the reference's contact state, plus low horizontal foot speed while
        the reference foot is down. Evaluation keeps ``w_contact = 0``, i.e. the paper's reward."""
        c, m = self.cfg, self.motion
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rp, rR = m.root_p[f], m.root_R[f]
        # reference bodies re-anchored to the robot's root (xy + yaw), as in BeyondMimic
        dR = yaw_mat(mm3(R, rR.transpose(-1, -2)))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + mv3(dR.unsqueeze(-3), m.body_p[f] - rp.unsqueeze(-2))
        ref_bR = mm3(dR.unsqueeze(-3), m.body_R[f])
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        def track(c):
            return (
                torch.exp(-e_pos / c.body_pos_std**2)
                + torch.exp(-e_ori / c.body_ori_std**2)
                + torch.exp(-e_lv / c.body_linvel_std**2)
                + torch.exp(-e_av / c.body_angvel_std**2)
                + 0.5 * torch.exp(-e_rp / c.root_pos_std**2)
                + 0.5 * torch.exp(-e_ro / c.root_ori_std**2)
            )

        r_track = track(c)
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r = (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim) * CONTROL_DT
        if w_contact > 0:
            fz, rz = bp[..., self.feet, 2], m.body_p[f][..., self.feet, 2]
            c_sim = torch.sigmoid((self.kin.contact_height - fz) / 0.005)
            rc = getattr(m, "ref_contact", None)  # MotionLib.set_ref_contact, else the robot's fixed contact height
            c_ref = rc[f] if rc is not None else (rz < self.kin.contact_height).float()
            slip = (bv[..., self.feet, :2] ** 2).sum(-1)
            r_contact = (1 - (c_sim - c_ref) ** 2) + c_ref * torch.exp(-slip / 0.3**2)
            r = r + w_contact * r_contact.mean(-1) * CONTROL_DT
        if terms:
            return r, {
                "track": r_track * CONTROL_DT,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "action_rate": r_rate,
                "joint_limit": r_lim,
            }
        return r

    def termination(self, S, f) -> torch.Tensor:
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    def critic_features(self, S, f, xi) -> torch.Tensor:
        """Critic input: invariant state, goal, sim context, and robot-vs-reference errors."""
        m = self.motion
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        Ht = H.transpose(-1, -2)
        rp, rR = m.root_p[f], m.root_R[f]
        bp = self.kin.forward(p, R, v, w, q, qd)[0]
        dR = yaw_mat(mm3(R, rR.transpose(-1, -2)))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + mv3(dR.unsqueeze(-3), m.body_p[f] - rp.unsqueeze(-2))
        e_body = mv3(Ht.unsqueeze(-3), ref_bp - bp).flatten(-2)
        e_root = mv3(Ht, rp - p)
        rel = mat_to_6d(mm3(R.transpose(-1, -2), rR))
        return torch.cat((invariant_features(S), m.q[f] - q, e_body, e_root, rel, xi), -1)
