"""Export a PartyMimic policy checkpoint to an ONNX with raw sim2real inputs (same design as export_onnx.py).

PartyMimic's policy sees, per history step, [joint pos - default, joint vel, root angular velocity (body),
projected gravity, previous action, 6D of R_root^T R_ref_root(f)] and a goal of K future reference frames
f+1..f+K: [heading-frame root displacement, heading-frame 6D orientation, root linear / angular velocity in the
frame's own root frame, joint pos, 0.1 * joint vel] (partymimic.motion.MotionLib.goal). The graph rebuilds both
from raw inputs; reference velocities are central differences at 50 Hz, as MotionLib computes them:
    proprio          [H*93]      supermimic.proprio_history
    root_quat_hist   [H*8]       supermimic.root_quat_history: robot root quat (motion frame), reference root quat
    ref_joint_pos    [(K+2)*29]  reference joint pos at frames f..f+K+1
    ref_root         [(K+2)*7]   reference root position + wxyz quaternion at frames f..f+K+1
A checkpoint trained with --interface sparse5 (DeploySparse5) takes the reference as keypoint poses instead and builds
the goal (motion.MotionLib.goal) and the keypoint-error block (mdp.MDP.kp_err) from them:
    proprio          [H*P]       supermimic.proprio_history
    root_quat_hist   [H*8]       supermimic.root_quat_history
    ref_kp           [(W+1)*6*7] supermimic.ref_bodies: world position + wxyz quaternion of pelvis, head, left wrist,
                                 right wrist, left ankle, right ankle (motion.sparse5_bodies) at frames f..f+W
Run with PartyMimic's venv and src on PYTHONPATH:
    PYTHONPATH=<partymimic>/src <partymimic>/.venv/bin/python export_deploy_onnx.py <ckpt.pt> <out.onnx>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import partymimic.networks as networks
from partymimic.kinematics import Kinematics
from partymimic.mdp import MDP, obs_dim, obs_err_dim
from partymimic.motion import CONTROL_DT, MotionLib, load_npz
from partymimic.networks import Policy, policy_kwargs
from partymimic.robot import get_robot
from partymimic.rotations import mat_to_6d, mat_to_quat, mm3, mv3, quat_to_mat, quat_to_rotvec, so3_log, yaw_mat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_yaml import build_blink, build_blink_robot, build_sparse5  # noqa: E402

SPEC = None  # the checkpoint's robot (load)
P = 93  # proprio entries per step before the 6D relative orientation: 3 J + 6


def _qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)


class DeployBlink(nn.Module):
    def __init__(self, pi: Policy, hist: int, K: int, J: int):
        super().__init__()
        self.pi, self.H, self.K, self.J = pi, hist, K, J
        self.register_buffer("conj", torch.tensor([1.0, -1.0, -1.0, -1.0]))

    def obs(self, proprio, root_quat_hist):
        qh = root_quat_hist.view(self.H, 8)
        rel = mat_to_6d(mm3(quat_to_mat(qh[:, :4]).transpose(-1, -2), quat_to_mat(qh[:, 4:])))
        return torch.cat((proprio.view(self.H, P), rel), -1).unsqueeze(0)

    def goal(self, ref_joint_pos, ref_root):
        K = self.K
        jp = ref_joint_pos.view(K + 2, self.J)
        rr = ref_root.view(K + 2, 7)
        p, q = rr[:, :3], rr[:, 3:]
        R = quat_to_mat(q)
        Rf = R[1 : K + 1]
        Rft = Rf.transpose(-1, -2)
        v = (p[2:] - p[:-2]) / (2 * CONTROL_DT)
        w = quat_to_rotvec(_qmul(q[2:], q[:-2] * self.conj)) / (2 * CONTROL_DT)
        Ht = yaw_mat(R[0]).transpose(-1, -2).unsqueeze(0)
        dp = mv3(Ht, p[1 : K + 1] - p[0:1])
        rot = mat_to_6d(mm3(Ht, Rf))
        qd = (jp[2:] - jp[:-2]) / (2 * CONTROL_DT)
        return torch.cat((dp, rot, mv3(Rft, v), mv3(Rft, w), jp[1 : K + 1], qd * 0.1), -1).reshape(1, -1)

    def forward(self, proprio, root_quat_hist, ref_joint_pos, ref_root):
        return self.pi(self.obs(proprio, root_quat_hist), self.goal(ref_joint_pos, ref_root)).reshape(self.J)


class DeploySparse5(nn.Module):
    """--interface sparse5 policy on the keypoint poses ref_kp of frames f..f+W (W the farthest goal frame); the error
    block is computed from the newest proprio step (joint encoders + default pose) and robot root orientation."""

    def __init__(self, pi: Policy, hist: int, K: int, J: int, kin: Kinematics, kp: list[int], q0: list[float]):
        super().__init__()
        self.pi, self.H, self.K, self.W, self.J, self.kin, self.kp = pi, hist, K, K, J, kin, kp
        self.register_buffer("q0", torch.tensor(q0, dtype=torch.float32))
        self.register_buffer("sel", torch.arange(1, K + 1))

    def _kp(self, ref_kp):
        x = ref_kp.view(self.W + 1, len(self.kp), 7)
        return x[..., :3], quat_to_mat(x[..., 3:])

    def goal(self, ref_kp):
        p, R = self._kp(ref_kp)
        Ht = yaw_mat(R[0, 0]).transpose(-1, -2)
        pf = mv3(Ht, p[self.sel] - p[0, 0])  # (K,6,3)
        rf = mat_to_6d(mm3(Ht, R[self.sel]))  # (K,6,6)
        return torch.cat((pf[:, 0], rf[:, 0], rf[:, 1], torch.cat((pf[:, 2:], rf[:, 2:]), -1).flatten(-2)), -1).reshape(1, -1)

    def err(self, proprio, root_quat_hist, ref_kp):
        q = proprio.view(self.H, P)[-1, : self.J] + self.q0
        R = quat_to_mat(root_quat_hist.view(self.H, 8)[-1, :4])
        bp, bR = self.kin.pose(mm3(yaw_mat(R).transpose(-1, -2), R), q, self.kp[1:])
        p, Rk = self._kp(ref_kp)
        Ht = yaw_mat(Rk[0, 0]).transpose(-1, -2)
        rp, rR = mv3(Ht, p[0, 1:] - p[0, 0]), mm3(Ht, Rk[0, 1:])
        e_R = so3_log(mm3(rR.transpose(-1, -2), bR))
        return torch.cat(((bp - rp)[1:].flatten(), e_R[1:].flatten(), e_R[0]))

    def obs(self, proprio, root_quat_hist, ref_kp):
        qh = root_quat_hist.view(self.H, 8)
        rel = mat_to_6d(mm3(quat_to_mat(qh[:, :4]).transpose(-1, -2), quat_to_mat(qh[:, 4:])))
        e = self.err(proprio, root_quat_hist, ref_kp).expand(self.H, -1)  # the policy reads the newest step's only
        return torch.cat((proprio.view(self.H, P), rel, e), -1).unsqueeze(0)

    def forward(self, proprio, root_quat_hist, ref_kp):
        return self.pi(self.obs(proprio, root_quat_hist, ref_kp), self.goal(ref_kp)).reshape(self.J)


def load(ckpt: str) -> tuple[Policy, dict]:
    global SPEC, P
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    SPEC = get_robot(a.get("robot", "g1"))
    P = 3 * len(SPEC.joint_names) + 6
    if a.get("wm_contact", 0) or a.get("pi_arch", "mlp") != "mlp":
        raise SystemExit("unsupported policy architecture")
    widths = lambda key, default: tuple(int(x) for x in a.get(key, default).split(","))  # noqa: E731
    J = len(SPEC.joint_names)
    K = a["goal_frames"]
    interface = a.get("interface", "joint")
    from partymimic.motion import SPARSE5_FRAME

    goal_dim = K * SPARSE5_FRAME if interface == "sparse5" else K * (15 + 2 * J)
    pi = Policy(obs_dim(J, interface), goal_dim, J, hist=a["hist"], enc_hidden=widths("enc_hidden", "2048,1024,512,256"),
                dec_hidden=widths("dec_hidden", "2048,1024,512,256"), **policy_kwargs(a, SPEC))
    pi.load_state_dict(ck["pi"])
    return pi.eval(), a


def kp_window(motion: MotionLib, f: int, W: int) -> torch.Tensor:
    """Raw ref_kp of frame f: keypoint poses of frames f..f+W clamped to the clip like sim2real's slicing."""
    end = int(motion.clip_end[motion.clip_id[f]])
    idx = torch.clamp(torch.arange(f, f + W + 1), max=end)
    return torch.cat((motion.kp_p[idx], mat_to_quat(motion.kp_R[idx])), -1).reshape(-1)


def export_sparse5(pi: Policy, args: dict, motion: MotionLib, kin: Kinematics, out: Path) -> None:
    J, K, H = len(SPEC.joint_names), args["goal_frames"], args["hist"]
    q0 = SPEC.default_joint_pos_list()
    dep = DeploySparse5(pi, H, K, J, kin, motion.kp, q0).eval()
    mdp = MDP(kin, motion, torch.tensor(q0), 0)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        frames = torch.randint(motion.num_frames, (200,), generator=g)
        geo = max((dep.goal(kp_window(motion, f, dep.W)) - motion.goal(torch.tensor([f]), K)).abs().max().item()
                  for f in frames.tolist())  # fmt: skip
        err_max, pol_err = 0.0, 0.0
        for i, f in enumerate(frames[:20].tolist()):
            rk = kp_window(motion, f, dep.W)
            prop = torch.randn(H * P, generator=g) * 0.3
            rq = torch.nn.functional.normalize(torch.randn(H, 4, generator=g), dim=-1)
            rq = torch.cat((rq, mat_to_quat(motion.root_R[f]).expand(H, 4)), -1).reshape(-1)
            Rr = quat_to_mat(rq.view(H, 8)[:, :4])
            q = prop.view(H, P)[-1, :J] + dep.q0
            e_ref = mdp.kp_err(Rr[-1:], q[None], torch.tensor([f]))[0]
            err_max = max(err_max, (dep.err(prop, rq, rk) - e_ref).abs().max().item())
            rel_ref = mat_to_6d(mm3(Rr.transpose(-1, -2), motion.root_R[f]))
            obs_ref = torch.cat((prop.view(H, P), rel_ref, e_ref.expand(H, -1)), -1).unsqueeze(0)
            pol_err = max(pol_err, (dep(prop, rq, rk) - pi(obs_ref, motion.goal(torch.tensor([f]), K))[0]).abs().max().item())
    inputs = (prop, rq, rk)
    names = ["proprio", "root_quat_hist", "ref_kp"]
    torch.onnx.export(dep, inputs, str(out), input_names=names, output_names=["action"], opset_version=17, dynamo=False)
    np.savez(out.with_suffix(".check.npz"), **{n: x.numpy() for n, x in zip(names, inputs)},
             action=dep(*inputs).detach().numpy())
    kp_names = [SPEC.tracked_bodies[i] for i in motion.kp]
    cfg = build_sparse5(out.name, dep.W, SPEC, kp_names, list(range(1, K + 1)), obs_err_dim("sparse5"))
    out.with_suffix(".yaml").write_text(__import__("yaml").safe_dump(cfg, sort_keys=False))
    print(json.dumps({"onnx": str(out), "robot": SPEC.name, "interface": "sparse5", "future": K, "window": dep.W,
                      "history": H, "keypoints": kp_names, "geometry_max_err": geo, "kp_err_max_err": err_max,
                      "policy_vs_training_max_err": pol_err, "inputs": names}))


def window(motion: MotionLib, f: int, K: int):
    """Raw ref_joint_pos / ref_root of frame f, frames clamped to the clip like sim2real's slicing."""
    end = int(motion.clip_end[motion.clip_id[f]])
    idx = torch.clamp(torch.arange(f, f + K + 2), max=end)
    from partymimic.rotations import mat_to_quat

    root = torch.cat((motion.root_p[idx], mat_to_quat(motion.root_R[idx])), -1)
    return motion.q[idx].reshape(-1), root.reshape(-1)


def check_geometry(dep: DeployBlink, motion: MotionLib, n: int = 200) -> float:
    """Graph goal against MotionLib.goal on frames at least two away from the clip end (where MotionLib switches to
    one-sided differences)."""
    g = torch.Generator().manual_seed(0)
    ok = (motion.clip_end[motion.clip_id] - torch.arange(motion.num_frames)) > dep.K + 1
    ok &= torch.arange(motion.num_frames) > motion.clip_start[motion.clip_id]
    frames = ok.nonzero().flatten()
    frames = frames[torch.randint(len(frames), (n,), generator=g)]
    err = 0.0
    for f in frames.tolist():
        jp, rr = window(motion, f, dep.K)
        ref = motion.goal(torch.tensor([f]), dep.K)
        err = max(err, (dep.goal(jp, rr) - ref).abs().max().item())
    return err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    a = ap.parse_args()
    networks.AMP_DTYPE = None
    pi, args = load(a.ckpt)
    J, K, H = len(SPEC.joint_names), args["goal_frames"], args["hist"]
    sparse5 = args.get("interface", "joint") == "sparse5"
    dep = None if sparse5 else DeployBlink(pi, H, K, J).eval()

    kin = Kinematics(SPEC, "cpu")
    # a few clips of the training library are enough for the checks
    md = Path(args.get("motion_dir") or SPEC.motion_dir)
    if not md.is_dir():  # trained on another machine
        md = Path(os.environ.get("MOTION_DIR", SPEC.motion_dir))
    files, loader = sorted(md.glob(SPEC.motion_glob))[:8], SPEC.load_motion
    if not files:
        files, loader = sorted(md.glob("*.npz"))[:8], load_npz
    if not files:
        raise SystemExit(f"no motions in {md} for the export checks; set MOTION_DIR")
    motion = MotionLib([str(f) for f in files], kin, "cpu", loader)
    if sparse5:
        motion.set_interface("sparse5", SPEC.tracked_bodies, SPEC.keypoint_bodies)
        export_sparse5(pi, args, motion, kin, Path(a.out))
        return
    with torch.no_grad():
        geo = check_geometry(dep, motion)
        # relative orientation and the full policy against Policy(obs_hist, MotionLib.goal)
        g = torch.Generator().manual_seed(1)
        c = min(3, len(motion.names) - 1)
        f = min(int(motion.clip_start[c]) + 100, int(motion.clip_end[c]))
        jp, rr = window(motion, f, K)
        prop = torch.randn(H * P, generator=g) * 0.3
        rq = torch.nn.functional.normalize(torch.randn(H, 4, generator=g), dim=-1)
        from partymimic.rotations import mat_to_quat

        rq = torch.cat((rq, mat_to_quat(motion.root_R[f]).expand(H, 4)), -1).reshape(-1)
        Rr = quat_to_mat(rq.view(H, 8)[:, :4])
        rel_ref = mat_to_6d(mm3(Rr.transpose(-1, -2), motion.root_R[f]))
        obs_ref = torch.cat((prop.view(H, P), rel_ref), -1).unsqueeze(0)
        pol_err = (dep(prop, rq, jp, rr) - pi(obs_ref, motion.goal(torch.tensor([f]), K))[0]).abs().max().item()

    inputs = (prop, rq, jp, rr)
    names = ["proprio", "root_quat_hist", "ref_joint_pos", "ref_root"]
    out = Path(a.out)
    torch.onnx.export(dep, inputs, str(out), input_names=names, output_names=["action"], opset_version=17,
                      dynamo=False)
    np.savez(out.with_suffix(".check.npz"), **{n: x.numpy() for n, x in zip(names, inputs)},
             action=dep(*inputs).detach().numpy())
    cfg = build_blink(out.name, K) if SPEC.name in ("g1", "g1mj") else build_blink_robot(out.name, K, SPEC)
    out.with_suffix(".yaml").write_text(__import__("yaml").safe_dump(cfg, sort_keys=False))
    print(json.dumps({"onnx": str(out), "robot": SPEC.name, "future": K, "history": H, "geometry_max_err": geo,
                      "policy_vs_training_max_err": pol_err, "inputs": names}))


if __name__ == "__main__":
    main()
