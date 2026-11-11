"""PartyMimic training: collect -> fit the world model -> BPTT policy + critic update (SuperMimic, Algorithm A.1), with
the defaults set to the PartyMimic recipe (README, "Recipe")."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ON = argparse.BooleanOptionalAction

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="g1mj", help="robot package under partymimic/ (g1mj, rp1, n3, g1)")
parser.add_argument("--motion_dir", type=str, default=None, help="defaults to the robot's motion directory")
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None, help="last.pt of a run to continue")
parser.add_argument("--init", type=str, default=None,
                    help="last.pt whose world model, policy and critic start this run (fine-tuning, e.g. on a few clips); "
                         "optimizers, learning-rate control and reset statistics start fresh")
parser.add_argument("--iterations", type=int, default=12000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--ckpt_minutes", type=float, default=0.0, help="also checkpoint every n minutes of training wall-clock")
parser.add_argument("--max_hours", type=float, default=None, help="stop after this much training wall-clock")
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--pipeline", action="store_true",
                    help="two GPUs: collection and the world model on cuda:0, the policy and critic update on cuda:1, "
                         "side by side; the policy update reads the replay buffer and world model of the previous iteration")
# ---- simulator, replay, imagination ----
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--interface", type=str, default="joint", choices=("joint", "sparse5"),
                    help="reference interface of the policy (motion.INTERFACES): joint = root motion + joint positions / "
                         "velocities per goal frame; sparse5 = pelvis / head / wrist / ankle poses per goal frame and an "
                         "explicit keypoint-error block for the decoder, with --kp_skip and --kp_aux on by default")
parser.add_argument("--kp_skip", action=ON, default=None,
                    help="sparse5 (default on): reference-action skip on an estimated reference, a = kp2q(goal) / alpha + dec(...), "
                         "kp2q an MLP from the goal to q_ref(f+1) - q0 fit to the motion library before training, frozen")
parser.add_argument("--kp_hidden", type=str, default="1024,1024,512", help="--kp_skip: hidden widths of kp2q")
parser.add_argument("--kp_steps", type=int, default=20000, help="--kp_skip: Adam steps (batch 8192) of the kp2q fit")
parser.add_argument("--kp_aux", type=float, default=None,
                    help="sparse5 (default 1.0): weight of an auxiliary loss on the policy, a linear head on the reference latent "
                         "predicting the goal frames' joint positions (rad, mean squared error)")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
# ---- reward and resets ----
parser.add_argument("--std_scale", type=float, default=0.35,
                    help="training reward only: scale of the body / root position and orientation kernel widths "
                         "(evaluation keeps SuperMimic's reward)")
parser.add_argument("--contact_reward", type=float, default=0.25,
                    help="weight of the contact-matching / anti-slip reward term in imagination (evaluation reward unchanged)")
parser.add_argument("--fail_reset", type=float, default=0.3,
                    help="share of resets sampled one bin before recent termination points (anti-forgetting of hard clips)")
parser.add_argument("--div_k", type=int, default=24,
                    help="reset sampling: the uniform share by motion diversity, k-means clusters of the bins' motion "
                         "features with mass ~ (bins in the cluster)^--div_tau (0 = by duration)")
parser.add_argument("--div_tau", type=float, default=0.0)
parser.add_argument("--ref_enc", type=str, default="mlp", choices=("mlp", "xattn"),
                    help="motion encoder: mlp = enc(goal); xattn = dynamics-conditioned cross-attention, a causal "
                         "Transformer over the observation history queries the goal frames (RGMT)")
parser.add_argument("--xattn_dim", type=int, default=128)
parser.add_argument("--xattn_heads", type=int, default=4)
# ---- world model ----
parser.add_argument("--wm_hidden", type=str, default="1024,1024,1024", help="world-model MLP widths")
parser.add_argument("--wm_structured", action=ON, default=True,
                    help="semi-implicit world model: velocity increments + position corrections on top of the trapezoidal drift")
parser.add_argument("--wm_mirror", type=float, default=0.5,
                    help="share of each world-model batch replaced by its left-right mirror image (physical symmetry; 0 = off)")
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=5)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--wm_opt", type=str, default="muon", choices=["adam", "muon"],
                    help="world-model optimizer: muon = Muon on the hidden weight matrices of the MLP, Adam elsewhere")
parser.add_argument("--wm_only_iters", type=int, default=0, help="world-model-only iterations at the start")
# ---- policy ----
parser.add_argument("--enc_hidden", type=str, default="2048,1024,512,256", help="motion-encoder MLP widths")
parser.add_argument("--dec_hidden", type=str, default="2048,1024,512,256", help="shared-decoder MLP widths")
parser.add_argument("--pi_ref_skip", action=ON, default=True,
                    help="reference-action skip connection, a = (q_ref(f+1) - q0) / alpha + network output")
parser.add_argument("--pi_batch", type=int, default=2048)
parser.add_argument("--pi_opt", type=str, default="muon", choices=["adam", "muon"],
                    help="policy optimizer: muon = Muon (Jordan et al. 2024) on the weight matrices of the encoder and "
                         "decoder at Adam's update RMS (same lr), Adam on vectors")
parser.add_argument("--pi_muon_io", action=ON, default=True,
                    help="--pi_opt muon on the first and last Linear of the policy MLPs too (else Adam there)")
parser.add_argument("--pi_beta1", type=float, default=0.9, help="Adam beta1 of the policy")
parser.add_argument("--pi_clip_value", type=float, default=0.1, help="element-wise policy gradient clip after the norm clip (<= 0: off)")
parser.add_argument("--pi_ema", type=float, default=0.995,
                    help="Polyak average of the policy weights (decay per update); checkpoints store the average as 'pi'")
parser.add_argument("--sil", type=float, default=1.0,
                    help="weight of self-imitation at the imagination starts: regress the policy action onto the "
                         "recorded (noisy) action where its recorded return G exceeds V(s_0), weighted by the advantage "
                         "(0 = off)")
parser.add_argument("--sil_wmax", type=float, default=5.0, help="--sil: cap of the advantage weight (mean |A| units)")
# ---- critic ----
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--v_gcons", type=float, default=0.5,
                    help="critic: weight of the value-gradient consistency loss, grad_s V(S_k) fit to the BPTT gradient of "
                         "the imagined return from S_k (k = 1..H-1, per-coordinate one-step-change scales; 0 = off)")
parser.add_argument("--v_gcons_n", type=int, default=4096, help="--v_gcons: states per critic minibatch")
parser.add_argument("--v_gcons_epochs", type=int, default=1, help="--v_gcons: in the minibatches of the first this many epochs")
# ---- learning rates ----
parser.add_argument("--lr", type=float, default=4e-4)
parser.add_argument("--lr_decay_iters", type=int, default=250_000, help="cosine decay of every learning rate")
parser.add_argument("--pi_lr_mult", type=float, default=4.0,
                    help="policy learning rate = --lr x this (world model and critic keep --lr)")
parser.add_argument("--lr_decay_ret", type=float, default=0.75,
                    help="progress-triggered decay of --pi_lr_mult (0 = off): once the logged training return (ep_ret_norm) "
                         "reaches this, the policy LR multiplier ramps linearly to 1 over --lr_decay_len iterations")
parser.add_argument("--lr_decay_len", type=int, default=300)
parser.add_argument("--lr_decay_by", type=int, default=1400,
                    help="with --lr_decay_ret: start the decay at this iteration at the latest, also if the training return "
                         "has not reached --lr_decay_ret (0 = no deadline)")
args = parser.parse_args()
if args.kp_skip is None:
    args.kp_skip = args.interface == "sparse5"
if args.kp_aux is None:
    args.kp_aux = 1.0 if args.interface == "sparse5" else 0.0
assert args.interface == "sparse5" or not (args.kp_skip or args.kp_aux > 0), "--kp_skip / --kp_aux: --interface sparse5 only"
assert not (args.resume and args.init), "--resume or --init"

import torch
import torch.nn as nn

from partymimic.buffer import ReplayBuffer
from partymimic.dr import XiLayout
from partymimic.env import PHYSICS_DT, TrackingSim, make_sim
from partymimic.kinematics import Kinematics
from partymimic.mdp import MDP, RewardCfg, inv_dim, unflatten_state
from partymimic.motion import CONTROL_DT, MotionLib
import partymimic.networks as networks
import partymimic.rotations as rotations
from partymimic.networks import Critic, Policy, WorldModel, check_pruned, policy_kwargs
from partymimic.mirror import Mirror
from partymimic.objective import actor_objective
from partymimic.mdp import flatten_state
from partymimic.robot import get_robot
from partymimic.sensing import ObsPipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
ldev = "cuda:1" if args.pipeline else dev  # the policy learner's GPU
os.makedirs(args.log_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

robot = get_robot(args.robot)
J = robot.num_joints
kin = Kinematics(robot, dev)
motion = MotionLib.from_dir(args.motion_dir or robot.motion_dir, robot, kin, dev)
motion.set_interface(args.interface, robot.tracked_bodies, robot.keypoint_bodies)
layout = XiLayout(robot, dev)
normalize_xi = layout.normalize
_rc, _s = RewardCfg(), args.std_scale
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), layout.dim, robot.sensing,
          RewardCfg(body_pos_std=_rc.body_pos_std * _s, body_ori_std=_rc.body_ori_std * _s,
                    root_pos_std=_rc.root_pos_std * _s, root_ori_std=_rc.root_ori_std * _s))
pipe = ObsPipeline(robot, mdp, layout, dev, PHYSICS_DT)
# every update and the collection step run as CUDA graphs: the sensing chain must be stateless
assert not pipe.stateful, f"{args.robot}: stateful sensing (filters, delays) is not supported"
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = motion.goal_dim(K)
CRITIC_IN = inv_dim(J) + mdp.critic_extra + GOAL_DIM

widths = lambda s: tuple(int(x) for x in s.split(","))  # noqa: E731
networks.RunningNorm.no_sync = True
rotations.ELEMENTWISE = True

wm = WorldModel(J, layout.dim, use_prev_action=robot.has_action_delay, hidden=widths(args.wm_hidden),
                structured=args.wm_structured, dt=CONTROL_DT).to(dev)  # fmt: skip
pi = Policy(mdp.obs_dim, GOAL_DIM, J, hist=Hh, enc_hidden=widths(args.enc_hidden), dec_hidden=widths(args.dec_hidden),
            **policy_kwargs(vars(args), robot)).to(dev)
Q0 = torch.tensor(robot.default_joint_pos_list(), device=dev)


def fit_kp2q(dst: networks.FrozenMLP) -> None:
    """--kp_skip: fit kp2q (goal -> q_ref(f+1) - q0) on all library frames but a held-out 5%, then copy it into dst."""
    g = torch.Generator(device=dev).manual_seed(args.seed)
    perm = torch.randperm(motion.num_frames, device=dev, generator=g)
    val, tr = perm[: motion.num_frames // 20], perm[motion.num_frames // 20 :]
    data = lambda f: (motion.goal(f, K), motion.q[motion.future_frames(f, K)[:, 0]] - Q0)  # noqa: E731
    X0 = motion.goal(tr[torch.randint(len(tr), (65536,), device=dev, generator=g)], K)
    mu, sd = X0.mean(0), X0.std(0).clamp(min=1e-3)
    net = networks.mlp(GOAL_DIM, list(widths(args.kp_hidden)), J).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t0 = time.time()
    for step in range(args.kp_steps):
        for grp in opt.param_groups:
            grp["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * step / args.kp_steps))
        X, Y = data(tr[torch.randint(len(tr), (8192,), device=dev, generator=g)])
        loss = ((net((X - mu) / sd) - Y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        se, se0 = torch.zeros(J, device=dev), torch.zeros(J, device=dev)
        for s in range(0, len(val), 65536):
            X, Y = data(val[s : s + 65536])
            se += ((net((X - mu) / sd) - Y) ** 2).sum(0)
            se0 += (Y**2).sum(0)
    rms, rms0 = (se / len(val)).sqrt(), (se0 / len(val)).sqrt()
    print(f"--kp_skip: kp2q fit in {time.time() - t0:.0f} s, held-out joint RMS {rms.mean():.4f} rad (max {rms.max():.4f}; "
          f"default pose {rms0.mean():.4f})", flush=True)
    dst.load_from(net, mu, sd)


if args.kp_skip:
    fit_kp2q(pi.kp2q)
pi_l = copy.deepcopy(pi).to(ldev) if args.pipeline else pi  # --pipeline: the trained policy; pi acts with its copy
critic = Critic(CRITIC_IN).to(ldev)
assert not args.lr_decay_by or args.lr_decay_ret > 0, "--lr_decay_by needs --lr_decay_ret"
# the learning rate is a device tensor (filled every iteration) and Adam keeps its step on the GPU, so the captured
# optimizer steps follow the schedule
adam_kw = lambda d: {"fused": True, "capturable": True, "lr": torch.tensor(args.lr, device=d)}  # noqa: E731
from partymimic.optim import FlatAdam, MuonAdam

if args.wm_opt == "muon":
    opt_wm = MuonAdam(wm, [wm.net], lr=args.lr, betas=(0.9, 0.999), fused=True, capturable=True, flat_adam=True)
else:
    opt_wm = torch.optim.Adam(wm.parameters(), **adam_kw(dev))
if args.pi_opt == "muon":
    opt_pi = MuonAdam(pi_l, [pi_l.enc, pi_l.dec], lr=args.lr, betas=(args.pi_beta1, 0.95), fused=True, capturable=True,
                      io=args.pi_muon_io)
else:
    opt_pi = torch.optim.Adam(pi_l.parameters(), betas=(args.pi_beta1, 0.95), **adam_kw(ldev))
NONFINITE = torch.zeros((), dtype=torch.long, device=ldev)  # skipped non-finite policy updates
pi_avg = torch.optim.swa_utils.AveragedModel(pi_l, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.pi_ema),
                                             use_buffers=True) if args.pi_ema > 0 else None  # fmt: skip
opt_v = FlatAdam(critic.parameters(), args.lr)

GC_ON = args.v_gcons > 0
SIL_DIAG = torch.zeros(2, device=ldev)  # --sil: share of starts with G > V, self-imitation loss
GC_DIAG = torch.zeros(2, device=ldev)  # --v_gcons: relative squared error of grad_s V, cosine to the target
KP_DIAG = torch.zeros(1, device=ldev)  # --kp_aux: RMS error (rad) of the latent's joint-position prediction
ALPHA = torch.tensor(robot.action_scale_list(), device=ldev)  # action units -> joint-setpoint radians

start_it, wall_offset, sim_steps = 0, 0.0, 0


def load_opt_states(ck: dict):
    """Optimizer states of a checkpoint; the device learning-rate tensors are kept (filled)."""
    lr_t = {id(o): [g["lr"] for g in o.param_groups] for o in (opt_wm, opt_pi, opt_v)}
    opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    for o in (opt_wm, opt_pi, opt_v):
        for g, lr0 in zip(o.param_groups, lr_t[id(o)]):
            if torch.is_tensor(lr0) and g["lr"] is not lr0:
                lr0.fill_(float(g["lr"]))
                g["lr"] = lr0


INIT_BINS = None  # --resume: reset-sampler statistics, copied into the simulator once it exists
RESUME_CTRL = None  # --resume: the control state of last.pt, applied once the objects exist
if args.resume:
    # needs the full training state: last.pt. "pi" is the Polyak average with --pi_ema (what evaluation uses), so the
    # live policy comes from "pi_live" and the average from "pi_avg"
    ck = torch.load(args.resume, map_location=dev, weights_only=False)
    assert "opt_pi" in ck, f"--resume {args.resume}: no optimizer state (use last.pt)"
    check_pruned(ck.get("args", {}), what=args.resume)
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck.get("pi_live", ck["pi"])), critic.load_state_dict(ck["critic"])
    pi_l.load_state_dict(ck.get("pi_live", ck["pi"]))
    load_opt_states(ck)
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
    if pi_avg is not None:
        if ck.get("pi_avg") is not None:
            pi_avg.load_state_dict(ck["pi_avg"])
        else:  # checkpoint of the code before this fix: "pi" holds the average; continue the EMA from it
            pi_avg.module.load_state_dict(ck["pi"])
            pi_avg.n_averaged.fill_(1)
            print(f"--resume: no pi_avg in {args.resume}; EMA continued from the saved averaged policy 'pi'", flush=True)
    RESUME_CTRL = ck.get("ctrl")
    if RESUME_CTRL is not None:
        INIT_BINS = {k: v.to(dev) for k, v in RESUME_CTRL["env_bins"].items()}
    else:
        print(f"--resume: no control state in {args.resume}; LR trigger and reset statistics restart", flush=True)
    # a warm restart either way: the replay buffer, simulator state and random streams are not saved
    del ck
elif args.init:
    ck = torch.load(args.init, map_location=dev, weights_only=False)
    assert "critic" in ck, f"--init {args.init}: no critic (use last.pt)"
    check_pruned(ck.get("args", {}), what=args.init)
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck.get("pi_live", ck["pi"])), critic.load_state_dict(ck["critic"])
    pi_l.load_state_dict(ck.get("pi_live", ck["pi"]))
    if pi_avg is not None:
        pi_avg.module.load_state_dict(ck["pi"])
        pi_avg.n_averaged.fill_(1)
    print(f"--init: weights of {args.init} (it {ck['iteration']})", flush=True)
    del ck


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((J,), 3e-3), torch.full((J,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, a_prev, xi):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], a_prev[:, j], normalize_xi(xi[:, j]))
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


_FEET = robot.foot_bodies
FEET = [robot.tracked_bodies.index(b) for b in _FEET] if all(b in robot.tracked_bodies for b in _FEET) else []
mdp.feet = FEET
assert args.contact_reward == 0 or FEET, "--contact_reward needs tracked foot bodies"
motion.set_ref_contact(FEET)


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.inc_norm.std(), ERR_FLOOR)
    S1 = unflatten_state(s1)
    e = (wm.state_error(S_hat, S1) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1]


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g, z):
    a = pi(o_hist, g, z)
    S1 = wm.step(S, a, a_prev, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1


def _imag_goals(f):
    """Goals of the reference frames f .. f + H (they do not depend on the imagined states) and the policy's motion
    latents of the first H, each as one batched pass (None with --ref_enc xattn, whose latent depends on the state)."""
    fs = (f.unsqueeze(0) + torch.arange(HZ + 1, device=f.device).unsqueeze(1)).flatten()
    G = motion.goal(fs, K).view(HZ + 1, f.shape[0], -1)
    if args.ref_enc != "mlp":
        return G, [None] * HZ
    return G, pi.encode(G[:HZ].flatten(0, 1)).view(HZ, f.shape[0], -1)


def _reward_term(S, f, a, a_prev):
    return mdp.reward(S, f, a, a_prev, w_contact=args.contact_reward), mdp.soft_termination(S, f)


def _gae_targets(V, rews, dones):
    """GAE(lambda) returns over the imagined transitions; V is (H+1, B), bootstrapped from V[-1]."""
    R = V[-1]
    targets = []
    for t in range(HZ - 1, -1, -1):
        R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
        targets.append(R)
    return torch.stack(targets[::-1])


def _critic_loss(X, Y):
    return ((critic(X) - Y) ** 2).mean()


@torch.no_grad()
def real_returns(s, xi_n):
    """--sil: the discounted return the simulator recorded over the HZ transitions from each start (the
    training reward recomputed on the recorded states: the simulator stores it without the contact term), bootstrapped
    with the critic if the episode did not end; (B,) returns, (B,) 1 where the whole window is recorded, (B, J) the
    recorded action at the start."""
    w, ok = buffer.forward_from(s["age"], s["env"], HZ, ("s1", "a", "a_prev", "f", "done"))
    B = ok.shape[0]
    S1 = unflatten_state(w["s1"].flatten(0, 1))
    r, _ = reward_term(S1, (w["f"] + 1).flatten(), w["a"].flatten(0, 1), w["a_prev"].flatten(0, 1))
    alive = torch.cumprod(1 - w["done"].float(), 0)
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    f_end = w["f"][-1] + 1
    X_end = critic_input_c(unflatten_state(w["s1"][-1]), f_end, xi_n, motion.goal(f_end, K))
    G = (disc * c_t * r.view(HZ, B)).sum(0) + args.gamma**HZ * alive[-1] * critic(X_end)
    return G, ok.float(), w["a"][0]


def _wm_norm_data(S0, S1, a, a_prev, xi):
    return wm.inputs(S0, a, a_prev, normalize_xi(xi)), wm.targets(S0, S1), WorldModel.increments(S0, S1)


def _critic_input(S, f, xi_n, g):
    return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)


wm_rollout = torch.compile(_wm_rollout)
wm_err = torch.compile(_wm_err)
imag_step = torch.compile(_imag_step)
reward_term = torch.compile(_reward_term)
critic_input_c = torch.compile(_critic_input)
wm_norm_data_c = torch.compile(_wm_norm_data)
gae_targets_c = torch.compile(_gae_targets)
critic_loss_c = torch.compile(_critic_loss)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
mirror = Mirror(robot, layout.slices, dev) if args.wm_mirror > 0 else None


def _mirror_windows(w: dict, n: int) -> dict:
    """Replace the first n windows by their mirror images (states, actions, sim context)."""
    out = dict(w)
    ms = lambda x: flatten_state(mirror.state(unflatten_state(x)))  # noqa: E731
    for k, fn in (("s", ms), ("s1", ms), ("a", mirror.joints), ("a_prev", mirror.joints), ("xi", mirror.xi)):
        out[k] = torch.cat((fn(w[k][:n]), w[k][n:]))
    return out


def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    if mirror is not None:
        w = _mirror_windows(w, int(args.wm_batch * args.wm_mirror))
    S_hat = wm_rollout(w["s"], w["a"], w["a_prev"], w["xi"])
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


HOOK = os.environ.get("PM_HOOK")  # diagnostics: a script exec'd in these globals before the listed iterations
HOOK_AT = {int(x) for x in os.environ.get("PM_HOOK_AT", "").split(",") if x}


def sample_valid_starts(B: int):
    # fixed-size selection without reading the valid count on the host: stable-sort the valid candidates first (only
    # starts within H frames of a clip end are invalid, < 1% of a large library); fewer than B valid ones (a library of
    # a few short clips) are repeated cyclically
    s, h = buffer.sample_starts(2 * B, Hh)
    f = s["f"]
    ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
    order = torch.sort((~ok).to(torch.uint8), stable=True).indices
    j = torch.arange(B, device=f.device)
    nv = ok.sum().clamp(min=1)
    idx = order[torch.where(j < nv, j, j % nv)]
    return {k: v[idx] for k, v in s.items()}, {k: v[idx] for k, v in h.items()}


@torch.no_grad()
def ema_update():
    """The EMA step of ``AveragedModel.update_parameters`` (same lerp) without its host-side first-call test; the
    host-side counter n_averaged is advanced by the training loop."""
    avg = list(pi_avg.module.parameters()) + list(pi_avg.module.buffers())
    live = [t.detach() for t in list(pi.parameters()) + list(pi.buffers())]
    assert all(t.is_floating_point() for t in avg)
    torch._foreach_lerp_(avg, live, 1 - args.pi_ema)


class Graphed:
    """Run ``fn`` eagerly until ``ready()``; then eagerly on a side stream for ``warmup`` calls, capture it as a CUDA
    graph on the next call and replay it from then on. Each call is one update; outputs are cloned out of the graph pool."""

    def __init__(self, fn, ready, warmup: int = 3):
        self.fn, self.ready, self.warmup, self.calls, self.graph, self.out = fn, ready, warmup, 0, None, None

    def __call__(self):
        if self.graph is None and not self.ready():
            return self.fn()
        self.calls += 1
        if self.calls <= self.warmup:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                out = self.fn()
            torch.cuda.current_stream().wait_stream(side)
            return out
        if self.graph is None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # return the eager phase's cached blocks before the graph takes its own pool
            t0 = time.time()
            self.graph = torch.cuda.CUDAGraph()
            # a capture stream of the current device (torch.cuda.graph's default one is made once, on the first device)
            with torch.cuda.graph(self.graph, stream=torch.cuda.Stream()):
                self.out = self.fn()
            torch.cuda.synchronize()
            print(f"cuda_graph: captured {self.fn.__name__} in {time.time() - t0:.1f} s, "
                  f"{torch.cuda.memory_reserved() / 2**30:.1f} GB reserved", flush=True)  # fmt: skip
        self.graph.replay()
        return tuple(x.clone() for x in self.out)


def gated_step(opt, gn):
    """Skip a non-finite update without a host read: zero the gradients on the GPU and gate the optimizer step
    (parameters and all optimizer state unchanged); counted in the log column nonfinite."""
    finite = torch.isfinite(gn)
    NONFINITE.add_((~finite).long())
    for p in pi.parameters():
        if p.grad is not None:
            p.grad.copy_(torch.where(finite, p.grad, 0.0))
    if isinstance(opt, torch.optim.Optimizer):
        opt.found_inf = (~finite).float()  # the fused kernel skips the step and undoes the step count
        try:
            opt.step()
        finally:
            opt.found_inf = None
    else:
        opt.step(ok=finite)


def _rot_vee(M):
    """(..., 3, 3) -> (..., 3): the axial vector of M - M^T."""
    return torch.stack((M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0], M[..., 1, 0] - M[..., 0, 1]), -1)


def _gc_coords(g, R, scale):
    """--v_gcons: a state-shaped tuple as one vector times the per-coordinate scales; the rotation part as its pairing with
    body-frame rotations (tangent to SO(3))."""
    parts = (g[0], _rot_vee(R.transpose(-1, -2) @ g[1]), *g[2:])
    return torch.cat([p.flatten(1) for p in parts], -1) * scale


def _gc_scale(S_c, B):
    """--v_gcons: per-coordinate std of the one-step changes along the imagined windows (rotations as rotation vectors)."""
    xs = [x.view(HZ + 1, B, *x.shape[1:]) for x in S_c]
    d = (xs[0][1:] - xs[0][:-1], 0.5 * _rot_vee(xs[1][:-1].transpose(-1, -2) @ xs[1][1:]), *(x[1:] - x[:-1] for x in xs[2:]))
    return torch.cat([v.flatten(2) for v in d], -1).flatten(0, 1).std(0).clamp(min=1e-6)


def value_grad_targets(Ss, r, d, c_t, v_boot, S_all, f_all, A_flat, Ap_flat):
    """--v_gcons: grad_{S_k} G_k for k = 1..H-1, G_k the imagined return from S_k bootstrapped with V(S_H): the BPTT state
    gradients of the policy loss (retained on Ss) minus the direct terms of the reward and soft termination at S_k (they
    belong to the step into S_k), over gamma^k c_k. Returns the state-shaped targets ((H-1) B rows) and a mask c_k > 0.5."""
    H, B = r.shape
    n = (H - 1) * B
    with torch.no_grad():
        G, Gs = v_boot.detach().float(), []
        for t in range(H - 1, 0, -1):
            G = r[t].detach() + args.gamma * (1 - d[t].detach()) * G
            Gs.append(G)
        Gk = torch.stack(Gs[::-1])  # G_1 .. G_{H-1}
        ct = c_t.detach()
        disc = args.gamma ** torch.arange(H, device=dev, dtype=torch.float32).unsqueeze(-1)
        w_r = (disc * ct)[:-1].flatten()  # reward at S_{t+1}: gamma^t c_t
        w_d = (args.gamma * disc * ct)[:-1].flatten() * Gk.flatten()  # termination at S_{t+1}: gamma^{t+1} c_t G_{t+1}
        ck = (disc * ct)[1:]  # gamma^k c_k
    S_leaf = tuple(x[:n].detach().requires_grad_() for x in S_all)
    rl, dl = reward_term(S_leaf, f_all[:n], A_flat[:n].detach(), Ap_flat[:n].detach())
    loc = torch.autograd.grad((w_r * rl - w_d * dl).sum(), S_leaf, allow_unused=True)
    out = []
    for i in range(6):
        cs = torch.stack([Ss[k][i].grad if Ss[k][i].grad is not None else torch.zeros_like(Ss[k][i]) for k in range(H - 1)])
        cs = -B * cs.flatten(0, 1) - (loc[i] if loc[i] is not None else 0)
        out.append(cs / ck.flatten().clamp(min=1e-3).view(-1, *[1] * (cs.dim() - 1)))
    return tuple(out), (ct[1:] > 0.5).float().flatten()


def gcons_loss(S, f, xi_n, g, T, m, scale):
    """--v_gcons on a minibatch of states: a surrogate whose parameter gradient is that of the relative squared error of
    grad_s V (double backward through the critic MLP only: the critic-input Jacobian enters by central differences along
    the error's direction), the error and the masked mean cosine to the target."""
    S = tuple(x.detach().requires_grad_() for x in S)
    X = critic_input_c(S, f, xi_n, g)
    Xl = X.detach().requires_grad_()
    u = torch.autograd.grad(critic(Xl).sum(), Xl, create_graph=True)[0]
    gS = torch.autograd.grad(X, S, grad_outputs=u.detach(), allow_unused=True)
    gS = tuple((x if x is not None else torch.zeros_like(s)).detach().requires_grad_() for x, s in zip(gS, S))
    R = S[1].detach()
    zt = _gc_coords(T, R, scale)
    diff = _gc_coords(gS, R, scale) - zt
    err = (m * (diff**2).sum(-1)).sum() / (m * (zt**2).sum(-1)).sum().clamp(min=1e-12)
    e = torch.autograd.grad(err, gS)
    with torch.no_grad():
        eps = 0.05 / _gc_coords(e, R, 1 / scale).norm(dim=-1).clamp(min=1e-12)
        shift = lambda sg: tuple(x + sg * eps.view(-1, *[1] * (x.dim() - 1)) * ex for x, ex in zip(S, e))  # noqa: E731
        Je = (critic_input_c(shift(1), f, xi_n, g) - critic_input_c(shift(-1), f, xi_n, g)) / (2 * eps.unsqueeze(-1))
        zg = diff + zt
        cos = (m * (zg * zt).sum(-1) / (zg.norm(dim=-1) * zt.norm(dim=-1)).clamp(min=1e-12)).sum() / m.sum().clamp(min=1)
    return (u * Je).sum(), err.detach(), cos


CS = {}  # the policy step's imagined windows for the critic fit


def policy_step():
    s, h = sample_valid_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    with torch.no_grad():
        o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
    o_hist = mdp.add_noise(o_hist)
    o_hist0 = o_hist
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    G_all, Z_all = _imag_goals(f)
    g0, g = G_all[0], G_all[HZ]
    S0 = S
    Ss, As, Aps, Fs = [], [], [], []
    for t in range(HZ):
        Aps.append(a_prev)
        S, a_prev, o_hist, f = imag_step(S, o_hist, a_prev, f, xi, xi_n, G_all[t], Z_all[t])
        Ss.append(S), As.append(a_prev), Fs.append(f)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    f_all = flat(Fs)
    A_flat, Ap_flat = flat(As), flat(Aps)
    r, d = reward_term(S_all, f_all, A_flat, Ap_flat)
    r, d = r.view(HZ, B), d.view(HZ, B)
    v_boot = critic(critic_input_c(S, f, xi_n, g))
    # survival products as a loop (cumprod's backward reads the host)
    J, c_t, c = actor_objective(r, d, v_boot, args.gamma, loop_cumprod=True)
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    loss = -J.mean()
    if args.sil > 0:
        G_rec, rec_ok, a_rec = real_returns(s, xi_n)
        with torch.no_grad():
            A = (G_rec - critic(critic_input_c(S0, s["f"], xi_n, g0))) * rec_ok
            SIL_DIAG[0].copy_((A > 0).float().sum() / rec_ok.sum().clamp(min=1))
            wgt = torch.clamp(A / (A.abs().mean() + 1e-6), 0, args.sil_wmax)
        l_sil = (wgt * (((As[0] - a_rec) * ALPHA) ** 2).mean(-1)).mean()
        loss = loss + args.sil * l_sil
        SIL_DIAG[1].copy_(l_sil.detach())
    if args.kp_aux > 0:
        q_hat = pi.aux(pi.latent(o_hist0, G_all[0]))
        l_kp = ((q_hat - (motion.q[motion.future_frames(s["f"], K)] - Q0).flatten(1)) ** 2).mean()
        loss = loss + args.kp_aux * l_kp
        KP_DIAG[0].copy_(l_kp.detach().sqrt())
    if GC_ON:
        for St in Ss[:-1]:
            for x in St:
                x.retain_grad()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    if GC_ON:
        gc_T, gc_m = value_grad_targets(Ss, r, d, c_t, v_boot, S_all, f_all, A_flat, Ap_flat)
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    if args.pi_clip_value > 0:
        nn.utils.clip_grad_value_(pi.parameters(), args.pi_clip_value)
    gated_step(opt_pi, gn)
    if pi_avg is not None:
        if int(pi_avg.n_averaged) > 0:
            ema_update()
        else:
            pi_avg.update_parameters(pi)
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    CS.update(S0=S0, Ss=[tuple(x.detach() for x in St) for St in Ss], f0=s["f"], Fs=Fs, G_all=G_all, xi_n=xi_n,
              rews=rews, dones=dones, gc=(gc_T, gc_m) if GC_ON else None)  # fmt: skip
    return loss.detach(), (r_sum / HZ / 0.1).mean(), gn.detach(), c.mean().detach()


def critic_fit():
    """GAE(lambda) targets over the imagined transitions of the last policy step; minibatch fit of the critic."""
    S0, Ss, Fs, G_all, xi_n, rews, dones = (CS[k] for k in ("S0", "Ss", "Fs", "G_all", "xi_n", "rews", "dones"))
    B = S0[0].shape[0]
    v_loss = torch.zeros(())
    with torch.no_grad():
        S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i] for St in Ss])), 0).flatten(0, 1) for i in range(6))
        f_c = torch.cat((CS["f0"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
        g_c = G_all.flatten(0, 1)
        xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
        X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
    critic.norm.update(X)
    with torch.no_grad():
        targets = gae_targets_c(critic(X), rews, dones)  # (H, B)
    Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
    n = Xf.shape[0]
    mb = n // args.critic_minibatches
    if GC_ON:
        gc_T, gc_m = CS["gc"]
        gc_scale = _gc_scale(S_c, B)
        gc_S = tuple(x[B : HZ * B] for x in S_c)  # S_1 .. S_{H-1}
        gc_f, gc_xi, gc_g = f_c[B : HZ * B], xi_c[B : HZ * B], g_c[B : HZ * B]
    for ep in range(args.critic_epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(args.critic_minibatches):
            idx = perm[i * mb : (i + 1) * mb]
            v_loss = critic_loss_c(Xf[idx], Yf[idx])
            v_tot = v_loss
            if GC_ON and ep < args.v_gcons_epochs:
                gi = torch.randint(0, gc_m.shape[0], (min(args.v_gcons_n, gc_m.shape[0]),), device=dev)
                sur, gc_err, gc_cos = gcons_loss(tuple(x[gi] for x in gc_S), gc_f[gi], gc_xi[gi], gc_g[gi],
                                                 tuple(x[gi] for x in gc_T), gc_m[gi], gc_scale)
                v_tot = v_tot + args.v_gcons * sur
                GC_DIAG[0].copy_(gc_err), GC_DIAG[1].copy_(gc_cos)
            opt_v.zero_grad(set_to_none=True)
            v_tot.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_v.step()
    return (v_loss.detach(),)


def policy_update():
    pl, ir, gn, surv = policy_step()
    (vl,) = critic_fit()
    return pl, vl, ir, gn, surv


# ------------------------------------------------------------------------------------------------
sim_cfg = make_sim(robot)
# the simulator wrapper calls these once per control step in eager mode otherwise (FK + rotations: thousands of kernels)
mdp.reward, mdp.termination, mdp.obs = torch.compile(mdp.reward), torch.compile(mdp.termination), torch.compile(mdp.obs)
motion.goal = torch.compile(motion.goal)
env = TrackingSim(sim_cfg, args.num_envs, robot, motion, mdp, dev, gamma=args.gamma, fail_reset=args.fail_reset,
                  div_k=args.div_k, div_tau=args.div_tau)
PROFILE = os.environ.get("PM_PROFILE") == "1"  # time the physics substep and the phases of an iteration
PROF_T = {"collect": 0.0, "wm": 0.0, "pi": 0.0, "n": 0}
if PROFILE:
    for k in range(220):
        if k == 20:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(1.0 / 200)
    torch.cuda.synchronize()
    print(f"profile: physics substep {1000 * (time.perf_counter() - t0) / 200:.2f} ms for {args.num_envs} worlds "
          f"(write_data_to_sim + sim.step + scene.update)", flush=True)
if INIT_BINS is not None:  # --resume: the checkpoint's reset-sampler statistics (episodes already running keep their start)
    env.bin_G.copy_(INIT_BINS["bin_G"]), env.bin_fail.copy_(INIT_BINS["bin_fail"]), env.bin_w_avg.copy_(INIT_BINS["bin_w_avg"])
buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, J, layout.dim)
obs_hist = torch.zeros(args.num_envs, Hh, mdp.obs_dim, device=dev)


CG = {"calls": 0}  # the collection step: static tensors and the graph


@torch.no_grad()
def _collect_core():
    """One collection step incl. the episode ends (bin statistics, masked resets), on the static tensors of CG (the
    body of the collection graph)."""
    o = env.observe()
    fresh = env.ep_step == 0
    oh = CG["obs_hist"]
    oh_new = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(oh), torch.cat((oh[:, 1:], o.unsqueeze(1)), 1))
    g = motion.goal(env.f, K)
    pi.obs_norm.update(o)
    pi.goal_norm.update(g)
    a = pi(oh_new, g)
    a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
    out = env.step_core(a)
    buffer.add_device(out)
    S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
    wm_in, wm_out, wm_inc = wm_norm_data_c(S0, S1, out["a"], out["a_prev"], out["xi"])
    wm.in_norm.update(wm_in)
    wm.out_norm.update(wm_out)
    wm.inc_norm.update(wm_inc)
    oh.copy_(oh_new)
    # the simulator wrapper rebinds these per step: copy them back into the static tensors the graph reads
    CG["f"].copy_(env.f), CG["ep_step"].copy_(env.ep_step), CG["a_prev"].copy_(env.a_prev)
    env.f, env.ep_step, env.a_prev = CG["f"], CG["ep_step"], CG["a_prev"]
    core = {"done": out["done"], "term": out["term"], "f": out["f"], "r_mean": out["r"].mean() / 0.1}
    out = env.step_tail_masked(out)  # writes f, ep_step, a_prev in place
    n = out["n_done"]
    nan = torch.full((), float("nan"), device=dev)
    core.update(ep_len=torch.where(n > 0, out["ep_len_sum"] / n.clamp(min=1), nan),
                ep_ret=torch.where(n > 0, out["ep_ret_norm_sum"] / n.clamp(min=1), nan),
                term_sum=out["term"].float().sum(), done_sum=n)  # fmt: skip
    return core


@torch.no_grad()
def collect_graphed_step():
    import warp as wp
    from mjlab.sim.sim_data import TorchArray

    global obs_hist
    if CG["calls"] == 0:
        CG.update(obs_hist=obs_hist.clone(), f=env.f.clone(), ep_step=env.ep_step.clone(), a_prev=env.a_prev.clone())
        env.f, env.ep_step, env.a_prev = CG["f"], CG["ep_step"], CG["a_prev"]
    CG["calls"] += 1
    if CG["calls"] <= 3:  # warmup on the default stream (Warp's own stream orders with it implicitly)
        out = _collect_core()
    elif "graph" not in CG:
        torch.cuda.synchronize()
        t0 = time.time()
        CG["graph"] = torch.cuda.CUDAGraph()
        # mjlab writes simulator arrays on Warp's stream and launches Warp's own step graph, neither of which may enter a
        # torch capture: write on the capture stream, launch the step's kernels (Warp's capture_while solver loop becomes a
        # conditional node of this graph)
        setitem, env.sim.use_cuda_graph = TorchArray.__setitem__, False
        TorchArray.__setitem__ = lambda self, idx, value: self._tensor.__setitem__(idx, value)
        try:
            with torch.cuda.graph(CG["graph"], capture_error_mode="thread_local"):
                ws = wp.stream_from_torch(torch.cuda.current_stream())
                with wp.ScopedStream(ws, sync_enter=False, sync_exit=False):
                    wp.capture_begin(stream=ws, external=True)
                    try:
                        CG["out"] = _collect_core()
                    finally:
                        CG["wp_graph"] = wp.capture_end(stream=ws)  # keep: it holds the captured kernels' modules
        finally:
            TorchArray.__setitem__, env.sim.use_cuda_graph = setitem, True
        torch.cuda.synchronize()
        print(f"cuda_graph: captured one collection step in {time.time() - t0:.1f} s", flush=True)
        CG["graph"].replay()  # the capture itself does not run the step
        out = CG["out"]
    else:
        CG["graph"].replay()
        out = CG["out"]
    buffer.advance_host()
    obs_hist = CG["obs_hist"]
    return out


@torch.no_grad()
def collect(n_steps: int, stats: dict):
    global sim_steps
    for _ in range(n_steps):
        core = collect_graphed_step()
        sim_steps += args.num_envs
        stats["r"].append(core["r_mean"].clone())
        # every step appends; steps without an ended episode carry NaN means (skipped in the log)
        for k_s, k_c in (("ep_len", "ep_len"), ("ep_ret", "ep_ret"), ("term", "term_sum"), ("done", "done_sum")):
            stats[k_s].append(core[k_c].clone())


assert not env.push_enabled, "the collection graph: no pushes"
# the replay buffer keeps its fill level on the device, so the captured sampling follows it: capture at once
buffer.use_device_ptr(device_size=True)


def _to_device(obj, d: str):
    """A shallow copy of obj with its tensor attributes (and lists / tuples of tensors) on device d."""
    out = copy.copy(obj)
    for k, v in vars(obj).items():
        if torch.is_tensor(v):
            setattr(out, k, v.to(d))
        elif isinstance(v, (list, tuple)) and v and all(torch.is_tensor(x) for x in v):
            setattr(out, k, type(v)(x.to(d) for x in v))
    return out


def wire_dtypes(m: nn.Module, tensors: list) -> list:
    """--pipeline: the dtype each tensor crosses between the GPUs in: bf16 for the Linear weights and biases when the MLPs
    run under bf16 autocast (they only enter its bf16 matmuls, so the receiver computes the same), else their own."""
    lin = {id(p) for mod in m.modules() if isinstance(mod, nn.Linear) for p in mod.parameters(recurse=False)}
    bf16 = networks.AMP_DTYPE == torch.bfloat16
    return [torch.bfloat16 if bf16 and id(t) in lin else t.dtype for t in tensors]


_pad8 = lambda n: -n % 8  # noqa: E731  (every part starts 8-byte aligned, so it can be viewed as its dtype)


def pack(tensors: list, wire: list) -> torch.Tensor:
    parts = []
    for t, w in zip(tensors, wire):
        b = t.detach().to(w).reshape(-1).view(torch.uint8)
        parts += [b, b.new_zeros(_pad8(b.numel()))]
    return torch.cat(parts)


@torch.no_grad()
def unpack(b: torch.Tensor, tensors: list, wire: list):
    """Copy the bytes b (from pack) into tensors."""
    o = 0
    for t, w in zip(tensors, wire):
        n = t.numel() * w.itemsize
        t.copy_(b[o : o + n].view(w).view(t.shape))
        o += n + _pad8(n)


class Link:
    """--pipeline: a message from one GPU to the other through pinned host memory, on side streams of both GPUs, so the
    transfer overlaps both GPUs' work (no peer access between the GPUs; the PCIe link of one can be narrow)."""

    def __init__(self, src: str, dst: str):
        self.src, self.dst = src, dst
        self.s_src, self.s_dst = torch.cuda.Stream(src), torch.cuda.Stream(dst)
        self.host = self.buf = None
        self.arrived, self.consumed = torch.cuda.Event(), torch.cuda.Event()
        self.pending = None  # the size and metadata of a message sent and not yet received

    def send(self, b: torch.Tensor, meta=None):
        """b: uint8 on the source GPU, written by its current stream."""
        n = b.numel()
        if self.host is None or self.host.numel() < n:
            torch.cuda.synchronize(self.src), torch.cuda.synchronize(self.dst)
            self.host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
            self.buf = torch.empty(n, dtype=torch.uint8, device=self.dst)
        self.s_src.wait_stream(torch.cuda.current_stream(self.src))
        self.s_src.wait_event(self.arrived)  # the host buffer is free once the last message left it
        b.record_stream(self.s_src)
        with torch.cuda.stream(self.s_src):
            self.host[:n].copy_(b, non_blocking=True)
        self.s_dst.wait_stream(self.s_src)
        self.s_dst.wait_event(self.consumed)  # the destination buffer is free once the last message was read
        with torch.cuda.stream(self.s_dst):
            self.buf[:n].copy_(self.host[:n], non_blocking=True)
            self.arrived.record()
        self.pending = (n, meta)

    def recv(self, read):
        """read(bytes, meta) on the destination GPU's current stream, after the last message arrived; False if none."""
        if self.pending is None:
            return False
        n, meta = self.pending
        self.pending = None
        torch.cuda.current_stream(self.dst).wait_event(self.arrived)
        read(self.buf[:n], meta)
        self.consumed.record(torch.cuda.current_stream(self.dst))
        return True


LEARNER_FNS = ("sample_valid_starts", "real_returns", "_imag_step", "_imag_goals", "_reward_term", "_critic_input",
               "_critic_loss", "_gae_targets", "_gc_coords", "_gc_scale", "value_grad_targets", "gcons_loss",
               "gated_step", "ema_update", "policy_step", "critic_fit")  # fmt: skip


def learner_namespace():
    """--pipeline: the policy update's functions rebound to globals that name the learner GPU's copies of the world
    model, replay buffer, motion library and MDP (and the trained policy pi_l as pi); the collection and the world-model
    update keep this module's objects on cuda:0."""
    import types

    kin_l = Kinematics(robot, ldev)
    motion_l = _to_device(motion, ldev)
    motion_l.__dict__.pop("goal", None)  # the compiled method of the actor's copy
    motion_l.device = ldev
    motion_l.goal = torch.compile(motion_l.goal)
    mdp_l = _to_device(mdp, ldev)
    for k in ("reward", "termination", "obs"):
        mdp_l.__dict__.pop(k, None)
    mdp_l.kin, mdp_l.motion = kin_l, motion_l
    buffer_l = ReplayBuffer(args.buffer_steps, args.num_envs, ldev, J, layout.dim)
    buffer_l.use_device_ptr(device_size=True)
    L = dict(globals())
    L.update(pi=pi_l, Q0=Q0.to(ldev), wm=copy.deepcopy(wm).to(ldev), buffer=buffer_l, motion=motion_l, mdp=mdp_l, kin=kin_l,
             normalize_xi=_to_device(layout, ldev).normalize, dev=ldev)  # fmt: skip
    for name in LEARNER_FNS:
        f = globals()[name]
        inner = getattr(f, "__wrapped__", None)  # @torch.no_grad()
        g = inner or f
        g = types.FunctionType(g.__code__, L, name, g.__defaults__, g.__closure__)
        L[name] = torch.no_grad()(g) if inner is not None else g
    for name in ("imag_step", "reward_term", "critic_input_c", "gae_targets_c", "critic_loss_c"):
        L[name] = torch.compile(L[{"critic_input_c": "_critic_input", "gae_targets_c": "_gae_targets",
                                   "critic_loss_c": "_critic_loss"}.get(name, "_" + name)])  # fmt: skip
    return L


class Pipe:
    """--pipeline: in iteration k the actor (cuda:0) collects with the policy of update k - 1 and fits the world model,
    while the learner (cuda:1) updates the policy on the transitions and world model of iteration k - 1. Messages: the
    actor's new transitions, world model and observation / goal statistics to the learner; the policy to the actor, sent
    between the policy step and the critic fit."""

    def __init__(self, L: dict):
        self.buf_l, wm_l = L["buffer"], L["wm"]
        a_t = list(wm.parameters()) + list(wm.buffers()) + list(pi.buffers())
        self.a_src, self.a_dst = a_t, list(wm_l.parameters()) + list(wm_l.buffers()) + list(pi_l.buffers())
        self.a_wire = wire_dtypes(wm, list(wm.parameters())) + [t.dtype for t in a_t[len(list(wm.parameters())):]]
        self.p_src, self.p_dst = list(pi_l.parameters()), list(pi.parameters())
        self.p_wire = wire_dtypes(pi_l, self.p_src)
        self.to_l, self.to_a = Link(dev, ldev), Link(ldev, dev)

    def _actor_msg(self, n: int) -> torch.Tensor:
        """The n rows the actor wrote last, its world model and statistics, as bytes on cuda:0."""
        rows = (torch.arange(n, device=dev) + (buffer.ptr - n)) % buffer.T
        parts = []
        for x in buffer.data.values():
            b = x.index_select(0, rows).reshape(-1).view(torch.uint8)
            parts += [b, b.new_zeros(_pad8(b.numel()))]
        return torch.cat(parts + [pack(self.a_src, self.a_wire)])

    def _read_actor(self, b: torch.Tensor, meta):
        n, ptr, size = meta
        bl = self.buf_l
        rows = (torch.arange(n, device=ldev) + (ptr - n)) % bl.T
        o = 0
        for x in bl.data.values():
            k = n * x[0].numel() * x.element_size()
            x.index_copy_(0, rows, b[o : o + k].view(x.dtype).view(n, *x.shape[1:]))
            o += k + _pad8(k)
        unpack(b[o:], self.a_dst, self.a_wire)
        bl.ptr_t.fill_(ptr), bl.size_t.fill_(size)
        bl.ptr, bl.size = ptr, size

    def send_actor(self, n: int):
        n = min(n, buffer.T)
        self.to_l.send(self._actor_msg(n), (n, buffer.ptr, buffer.size))

    def recv_actor(self):
        with torch.cuda.device(ldev):
            self.to_l.recv(self._read_actor)

    def sync_actor(self, n: int):
        """The same message, copied at once (the warmup's transitions do not fit the links' buffers)."""
        n = min(n, buffer.T)
        self._read_actor(self._actor_msg(n).to(ldev), (n, buffer.ptr, buffer.size))
        torch.cuda.synchronize(ldev)

    def send_pi(self):
        with torch.cuda.device(ldev):
            self.to_a.send(pack(self.p_src, self.p_wire))

    def recv_pi(self):
        self.to_a.recv(lambda b, _: unpack(b, self.p_dst, self.p_wire))


wm_update_g = Graphed(wm_update, lambda: True)
_ready = lambda: int(pi_avg.n_averaged if pi_avg is not None else 1) > 0  # noqa: E731
if args.pipeline:
    PL = learner_namespace()
    PIPE = Pipe(PL)
    policy_step_g = Graphed(PL["policy_step"], _ready)
    # the critic fit reads the policy step's tensors: static once that step is a graph
    critic_fit_g = Graphed(PL["critic_fit"], lambda: policy_step_g.graph is not None)
else:
    policy_update_g = Graphed(policy_update, _ready)


stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
collect(args.warmup_steps, stats)  # the replay buffer is not checkpointed, so a resumed run refills it too
if args.pipeline:
    PIPE.sync_actor(args.warmup_steps)
log_path = os.path.join(args.log_dir, "train_log.csv")
log_f = open(log_path, "a")
if start_it == 0:
    log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr"
                ",pi_lr,pi_mult,nonfinite\n")  # fmt: skip
    log_f.flush()
LOG_EXTRA = "pi_lr" in open(log_path).readline()  # lr = base LR; pi_lr = policy LR (x pi_mult)
# periodic checkpoints hold only the deployed policy (what the evaluators read) and are written by one background writer
# (a failed write is re-raised in the training loop or at the end, at most 2 queued); last.pt is written at the end
CKPT_POOL, CKPT_PENDING = ThreadPoolExecutor(max_workers=1), []


def ckpt_wait(drain=False):
    while CKPT_PENDING and (drain or CKPT_PENDING[0].done() or len(CKPT_PENDING) > 2):
        CKPT_PENDING.pop(0).result()


def _save_atomic(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


t_start = time.time() - wall_offset
next_ckpt_s = wall_offset + args.ckpt_minutes * 60
t_log = time.time()
step_acc = 0.0
agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
# first iteration with a policy update (after --wm_only_iters); a resumed run is past it
POLICY_START = args.wm_only_iters if (start_it > 0 or args.wm_only_iters == 0) else None
DECAY_AT = [None]  # --lr_decay_ret: iteration of the trigger
if RESUME_CTRL is not None:
    DECAY_AT[0] = RESUME_CTRL["decay_at"]
    print(f"--resume: LR trigger at {DECAY_AT[0]}", flush=True)


for it in range(start_it, args.iterations):
    lr = lr_at(it)
    pi_mult = args.pi_lr_mult  # --lr_decay_ret: ramps to 1 after the trigger
    if DECAY_AT[0] is not None:
        pi_mult = args.pi_lr_mult + (1.0 - args.pi_lr_mult) * min(1.0, (it - DECAY_AT[0]) / max(args.lr_decay_len, 1))
    for o in (opt_wm, opt_pi, opt_v):
        for gparam in o.param_groups:
            if torch.is_tensor(gparam["lr"]):
                gparam["lr"].fill_(lr * (pi_mult if o is opt_pi else 1.0))
            else:
                gparam["lr"] = lr * (pi_mult if o is opt_pi else 1.0)
    if HOOK and it in HOOK_AT:
        exec(open(HOOK).read(), globals())
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    if PROFILE:
        torch.cuda.synchronize()
        _t = time.perf_counter()
    if args.pipeline:
        PIPE.recv_pi()
    collect(n, stats)
    if PROFILE:
        torch.cuda.synchronize()
        PROF_T["collect"] += time.perf_counter() - _t
        _t = time.perf_counter()
    for _ in range(args.wm_batches):
        l, e8 = wm_update_g()
    if PROFILE:
        torch.cuda.synchronize()
        PROF_T["wm"] += time.perf_counter() - _t
        _t = time.perf_counter()
    agg["wm"].append(l), agg["wm8"].append(e8)
    if POLICY_START is None and it + 1 >= args.wm_only_iters:
        POLICY_START = args.wm_only_iters
        print(f"policy updates start at iteration {POLICY_START}", flush=True)
        open(os.path.join(args.log_dir, "policy_start.txt"), "w").write(f"{POLICY_START}\n")
    if POLICY_START is not None and it >= POLICY_START:
        n_avg0 = int(pi_avg.n_averaged) if pi_avg is not None else 0
        with torch.cuda.device(ldev):
            if args.pipeline:
                PIPE.recv_actor()
                pl, ir, gn, surv = policy_step_g()
                PIPE.send_pi()
                (vl,) = critic_fit_g()
            else:
                pl, vl, ir, gn, surv = policy_update_g()
        if pi_avg is not None and int(pi_avg.n_averaged) == n_avg0:
            pi_avg.n_averaged += 1  # ema_update (and a graph replay) does not advance the host-side counter
        agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
    elif args.pipeline:
        PIPE.recv_actor()
    if args.pipeline:
        PIPE.send_actor(n)
    if PROFILE:
        torch.cuda.synchronize()
        PROF_T["pi"] += time.perf_counter() - _t
        PROF_T["n"] += 1
        if PROF_T["n"] % args.log_every == 0:
            k = PROF_T["n"]
            print(f"profile it {it + 1}: collect {1000 * PROF_T['collect'] / k:.1f} ms, world model "
                  f"{1000 * PROF_T['wm'] / k:.1f} ms, policy {1000 * PROF_T['pi'] / k:.1f} ms per iteration", flush=True)
            PROF_T.update(collect=0.0, wm=0.0, pi=0.0, n=0)
    if (it + 1) % args.log_every == 0:
        mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
        mean_nn = lambda x: mean([v for v in (float(v) for v in x) if v == v])  # noqa: E731  (NaN = no episode end)
        ips = args.log_every / (time.time() - t_log)
        t_log = time.time()
        wall_h = (time.time() - t_start) / 3600
        term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
        row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
               mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean_nn(stats["ep_len"]), mean_nn(stats["ep_ret"]), term_frac, ips, lr]  # fmt: skip
        extra = [lr * pi_mult, float(pi_mult), int(NONFINITE)] if LOG_EXTRA else []
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row + extra) + "\n")
        deadline = args.lr_decay_by > 0 and it + 1 >= args.lr_decay_by
        if args.lr_decay_ret > 0 and DECAY_AT[0] is None and (row[12] >= args.lr_decay_ret or deadline):
            DECAY_AT[0] = it + 1
            why = f"training return {row[12]:.3f} >= {args.lr_decay_ret}" if row[12] >= args.lr_decay_ret else f"deadline --lr_decay_by {args.lr_decay_by}"
            print(f"--lr_decay_ret: {why} at it {it + 1}: policy LR x{args.pi_lr_mult} -> x1 over {args.lr_decay_len} its", flush=True)
        log_f.flush()
        print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
              f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s"
              + (" | G>V {:.2f} sil {:.4f}".format(*SIL_DIAG.tolist()) if args.sil > 0 else "")
              + (" | gcons err {:.3f} cos {:.3f}".format(*GC_DIAG.tolist()) if GC_ON else "")
              + (" | kp_aux rms {:.4f}".format(*KP_DIAG.tolist()) if args.kp_aux > 0 else ""), flush=True)  # fmt: skip
        stats = {k: [] for k in stats}
        agg = {k: [] for k in agg}
    wall = time.time() - t_start
    out_of_time = args.max_hours is not None and wall >= args.max_hours * 3600
    timed = args.ckpt_minutes > 0 and wall >= next_ckpt_s
    if timed:
        next_ckpt_s += args.ckpt_minutes * 60
    final_ck = it + 1 == args.iterations or out_of_time
    if not final_ck and ((it + 1) % args.ckpt_every == 0 or timed):
        sd = pi_avg.module.state_dict() if pi_avg is not None else pi_l.state_dict()
        small = {"pi": {k: v.detach().to("cpu", copy=True) for k, v in sd.items()}, "iteration": it + 1,
                 "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                 "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        ckpt_wait()
        CKPT_PENDING.append(CKPT_POOL.submit(_save_atomic, small, path))
    elif (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations or timed or out_of_time:
        ck = {"wm": wm.state_dict(), "pi": pi_avg.module.state_dict() if pi_avg is not None else pi_l.state_dict(), "pi_live": pi_l.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
              "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
              "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
              "pi_avg": pi_avg.state_dict() if pi_avg is not None else None,
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        ck["ctrl"] = {"decay_at": DECAY_AT[0],
                      "env_bins": {"bin_G": env.bin_G.cpu(), "bin_fail": env.bin_fail.cpu(), "bin_w_avg": env.bin_w_avg.cpu()}}  # fmt: skip
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        small = {k: ck[k] for k in ("pi", "pi_live", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        torch.save(small, path + ".tmp")
        os.replace(path + ".tmp", path)
    if out_of_time:
        print(f"time budget of {args.max_hours} h reached at it {it + 1}", flush=True)
        break
ckpt_wait(drain=True)  # every background checkpoint written (a failed write raises here: non-zero exit)
CKPT_POOL.shutdown()
