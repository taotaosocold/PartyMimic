# PartyMimic

**General humanoid motion tracking in minutes on one GPU.** PartyMimic trains one whole-body tracking policy for a
41 h motion library on a single RTX 4090 and tracks motions it has never seen, on two humanoids (Unitree G1 and
RoboParty RP1). This repository is the method: the training code, the data recipe built only from public datasets,
the RP1 retargeting, the diversity analysis and the evaluation. Two trained checkpoints are included.

- **Fast:** 26 min of training on 1× RTX 4090 (24 GB).
- **Diverse data:** 41 h from LAFAN1, a diversity-selected 10k subset of BONES-SEED and KungfuAthlete.
- **Generalizes:** zero-shot on five held-out test sets (Loco, Mani, Dance, Ground, Extreme martial arts).
- **Precise:** 20.5 mm on the held-out manipulation clips (G1), the lowest error of every method we compared.
- **Two architectures:** an MLP reference encoder (default) or a cross-attention encoder (`--ref_enc xattn`).
- **Two modalities:** joint-angle references (default) or the pelvis and five keypoints (`--interface sparse5`).

PartyMimic is a model-based learner in the SuperMimic family. It fits an online MLP world model, trains the policy
by backpropagating through 32 imagined steps of that model, and uses a critic beyond the horizon. Project page:
[roboparty.github.io/PartyMimic](https://roboparty.github.io/PartyMimic/) ([`docs/index.html`](docs/index.html)),
with videos, the data-diversity analysis, the recipe ablation and the SuperMimic full-factor ablation.

![Mean held-out error against training compute](docs/static/hero.png)

## Results on held-out motions

Body position error (mm, lower is better), scored by the
[sim2real leaderboard](https://egalahad.github.io/sim2real/leaderboard/) sim2sim evaluator (MuJoCo, every clip from
its first frame). Loco, Mani, Dance and Ground are the MotionDecode splits; Extreme is the held-out 30% of
KungfuAthlete (whole source videos held out). Mean is the mean of the five.

| G1 | Training | Loco | Mani | Dance | Ground | Extreme | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| SONIC (official) | 21,000 GPU-h | 28.7 | 45.5 | 48.7 | 80.3 | 88.7 | 58.4 |
| Mimic Lite v1.1 (official) | 50 GPU-h | 25.0 | 28.2 | 43.3 | 145.2 | 95.7 | 67.5 |
| SuperMimic | 52 min | 34.9 | 34.3 | 55.5 | 119.2 | 95.5 | 67.9 |
| NablaMimic | 120 min | 32.9 | 30.3 | 46.9 | 107.8 | 81.6 | 59.9 |
| BlinkMimic | 59 min | 32.9 | 30.9 | 54.1 | 100.1 | 81.4 | 59.9 |
| XMimic | 50 min | 32.3 | 30.6 | 53.5 | 101.3 | 83.1 | 60.2 |
| **PartyMimic** (MLP) | **26 min** | 25.7 | **20.5** | 36.8 | 80.6 | 62.0 | 45.1 |
| **PartyMimic** (attention) | 41 min | 25.1 | 21.1 | **35.0** | **76.2** | **60.3** | **43.5** |

| RP1 | Training | Loco | Mani | Dance | Ground | Extreme | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| SuperMimic | 149 min | 34.8 | 29.1 | 53.4 | 162.5 | 90.2 | 74.0 |
| BlinkMimic | 68 min | 33.7 | 27.3 | 50.3 | 110.5 | 80.3 | 60.4 |
| XMimic | 59 min | 33.8 | 29.1 | 49.7 | 121.3 | 79.5 | 62.7 |
| **PartyMimic** (MLP) | **26 min** | 23.4 | 16.5 | 35.3 | 105.1 | 58.9 | 47.8 |
| **PartyMimic** (attention) | 42 min | **22.9** | **16.1** | **33.4** | **94.3** | **56.2** | **44.6** |
| PartyMimic (sparse5, MLP) | 29 min | 24.7 | 19.1 | 38.3 | 100.9 | 60.7 | 48.7 |

<sub>SuperMimic (paper hyperparameters), NablaMimic (official config), BlinkMimic (main preset), XMimic (its defaults)
and PartyMimic were all trained by us on the same library (G1: LAFAN1 + BONES-SEED subset + KungfuAthlete train
split; RP1: the same clips retargeted to RP1), each run on one RTX 4090; 2-seed means. Time is training wall-clock.
SONIC and Mimic Lite are official leaderboard checkpoints with other data and hardware. Completion rates and the
full protocol are on the [project page](docs/index.html#results).</sub>

## Checkpoints

Two checkpoints, the default recipe (MLP encoder, joint references), seed 0, 12,000 iterations:

| File | Robot | Training | Loco | Mani | Dance | Ground | Extreme | Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [`checkpoints/g1_mlp.pt`](checkpoints/g1_mlp.pt) | G1 (`--robot g1mj`) | 27 min | 25.5 | 20.6 | 36.7 | 82.6 | 61.9 | 45.5 |
| [`checkpoints/rp1_mlp.pt`](checkpoints/rp1_mlp.pt) | RP1 (`--robot rp1`) | 29 min | 23.3 | 16.5 | 35.5 | 102.3 | 59.2 | 47.4 |

Each holds the policy, world model and critic (no optimizer state). Export for the sim2real controller (sim2sim
and real robot), score, or fine-tune:

```bash
mkdir -p deploy && uv run python tools/export_deploy_onnx.py checkpoints/g1_mlp.pt deploy/g1_mlp.onnx  # + .yaml
SIM2REAL=<patched sim2real checkout> bash scripts/eval.sh checkpoints/g1_mlp.pt eval/g1_mlp g1    # see Evaluate
bash scripts/train.sh logs/g1_ft --robot g1mj --motion_dir <a few clips> --init checkpoints/g1_mlp.pt --iterations 2000
```

The export checks the ONNX against the PyTorch policy on a few training clips (at least 4). Without the training
library it uses the LAFAN1 clips for G1; for RP1 point `MOTION_DIR` at a directory of RP1 motion npz clips.

## Setup

Linux x86_64, an NVIDIA GPU with a CUDA 12.8 driver, and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:Roboparty/PartyMimic.git partymimic && cd partymimic
uv sync                    # mjlab, MuJoCo Warp, torch 2.9.0 (CUDA 12.8)
tools/download_data.sh     # G1 description + 40 LAFAN1 G1 clips, RP1 meshes + MJCF, G1 MJCF of the diversity analysis
```

`download_data.sh` gets the RP1 meshes from [Roboparty/rp1_description](https://github.com/Roboparty/rp1_description)
(pinned commit) and copies the capsule-collision MJCF of [`assets/rp1_28dof_mjlab`](assets/rp1_28dof_mjlab) next to
them (`data/rp1_28dof_mjlab/`), which is the model PartyMimic and the evaluator simulate.

**N3 V3.** `python tools/data/n3_assets.py` prepares `data/n3_29dof_mjlab/n3_29dof.xml` and its
meshes from the vendored `assets/n3_29dof_mjlab/` sources (also run by `download_data.sh`). Training
loads this `data/` copy, as for RP1. Robot parameters follow the N3 configuration in GR00T-WholeBodyControl;
the model, meshes and provenance are self-contained. See `assets/n3_29dof_mjlab/README.md`.

## Data

Every clip comes from a public dataset:

| Dataset | Used for | Source | License |
|---|---|---|---|
| LAFAN1, G1 retarget | training (40 clips) | [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) ([LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)) | CC BY-NC-ND 4.0 |
| BONES-SEED, G1 retarget | training (10,000 clips) | [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) (gated) | [BONES-SEED license](https://bones.studio/info/seed-license) |
| KungfuAthlete, G1 | training (635 clips), Extreme test (308 clips) | [LuluCao/KungfuAthleteBot](https://huggingface.co/datasets/LuluCao/KungfuAthleteBot), [NPCLEI/KungFuAthleteBot](https://github.com/NPCLEI/KungFuAthleteBot) | Apache-2.0 |
| MotionDecode | Loco / Mani / Dance / Ground tests (228 clips) | [CMRobot/MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode), as packaged by [sim2real](https://github.com/EGalahad/sim2real) | non-commercial access terms |

**G1 library.** `tools/data/build_library.sh g1` builds it from the sources (23.5 GB BONES-SEED download; accept
its terms on Hugging Face and log in with `hf auth login` or set `HF_TOKEN`):

1. LAFAN1: the 40 csv clips of `download_data.sh` to motion npz (`tools/data/prep_motions.py --format lafan`).
2. BONES-SEED: the 10,000 clips of [`manifests/bones_seed_div10k.txt`](manifests/bones_seed_div10k.txt), 120 Hz csv
   to 50 Hz npz. The list was selected by `tools/data/bones_subset.py` (below).
3. KungfuAthlete: `tools/data/kungfu.py qa` scores every smoothed G1 clip (floor offset, ground and self penetration,
   joint limits, speeds); `write` drops the failing clips, corrects the floor and writes the train split and the
   Extreme test set of [`manifests/kungfu_split.json`](manifests/kungfu_split.json) (a 30% test share per subset and
   style, drawn by whole source videos, so no test clip shares a video with a training clip).
4. All clips are linked into `data/motions/g1_mix`; the Extreme set is `data/testsets/extreme_g1`.

The selection of the BONES-SEED subset can be rerun (`feat` needs BONES-SEED's `metadata/seed_metadata_v004.parquet`,
`select` a GPU) and gives the same list:

```bash
uv run --with pyarrow python tools/data/bones_subset.py feat data/raw/bones data/raw/bones/feat.npz --meta <seed_metadata_v004.parquet>
uv run python tools/data/bones_subset.py select data/raw/bones/feat.npz div10k.txt --n 10000
```

It clusters 1-s windows of all non-mirror clips (without clips more than 3 cm below the floor or faster than
25 rad/s at a joint) and greedily picks clips that fill the clusters evenly, every content once before repeats.

**RP1 library.** The same clips, retargeted G1 → RP1 with the robot-to-robot IK of
[human-humanoid-tools](https://github.com/Roboparty/human-humanoid-tools) (commit `ed8b350`; `uv sync --extra all`
in its checkout):

```bash
HHT=<human-humanoid-tools checkout> P=8 GPUS="0 1" tools/data/build_library.sh rp1
```

It writes the G1 clips as csv (`tools/data/rp1_data.py g1csv`), retargets them (`tools/data/rp1_retarget.sh`: profile
`mimic`, Newton backend, 24 IK iterations, 50 Hz, the zero-pose calibration and the robot presets of
[`assets/hhtools_robots`](assets/hhtools_robots)), converts the RP1 csv to npz (`data/motions/rp1_mix`) and moves the
KungfuAthlete clips to the G1 floor height (`tools/data/rp1_floor.py`; R2R leaves some jump clips 2–5 cm above the
ground). The Extreme set is retargeted the same way (`data/testsets/extreme_rp1`). R2R takes about 8 s per clip per
process; `P` processes run in parallel over the GPUs in `GPUS`. The RP1 copy of the MotionDecode tests:
`HHT=... SIM2REAL=... tools/data/build_library.sh rp1_md` (into `$SIM2REAL/datasets/motiondecode_rp1`).

**Motion format.** `--motion_dir` takes a directory of `.npz` clips, each with `fps` (50), `joint_pos` `[T, J]` in
the robot's joint order (`src/partymimic/<robot>/`), and the pelvis pose as the first body of `body_pos_w`
`[T, B, 3]` and `body_quat_w` `[T, B, 4]` (wxyz). PartyMimic computes the other bodies by forward kinematics.

## Train

The defaults of `partymimic.train` are the PartyMimic recipe:

```bash
bash scripts/train.sh logs/g1_s0  --robot g1mj --motion_dir data/motions/g1_mix  --seed 0   # 12k iterations, ~26 min
bash scripts/train.sh logs/rp1_s0 --robot rp1  --motion_dir data/motions/rp1_mix --seed 0
bash scripts/train.sh logs/n3_s0  --robot n3   --motion_dir data/motions/n3/seed --seed 0
# two GPUs: collection + world model on cuda:0, policy + critic on cuda:1, overlapped (1.6x the iterations per second)
bash scripts/train.sh logs/g1_p0  --robot g1mj --motion_dir data/motions/g1_mix --seed 0 --pipeline
```

N3 reads NPZ clips recursively, including date subdirectories. Point `--motion_dir` at the selected
`seed/` directory, or link it to `data/motions/n3/seed`; `seed_unclassified/` stays separate. Named
N3_0905 short joint names and V3 joint names are supported. Existing DFS data is read by name without
another BFS permutation. The loader returns root pose and joint coordinates; PartyMimic recomputes
velocities and FK with its N3 V3 model. This does not retarget N3_0905 geometry to V3. Check the
reference motions' feet, joint limits and self-collisions before a full training run. `--interface sparse5`
uses the actual N3 head and hands; joint and sparse5 share the usual training and export commands.
N3 training results and hardware deployment are not part of the G1/RP1 benchmarks above; use
`partymimic.evaluate --robot n3` for this simulator, since the external `sim2real` evaluator needs its
own N3 robot support.

The MLP recipe fits a standard 24 GB RTX 4090: with only 24.1 GB free, G1 and RP1 runs kept their speed and peaked
at 22.6 and 23.4 GB. Checkpoints are written every 2,000 iterations (`ckpt_002000.pt`, ...) and `last.pt`.

**Options.**

- `--ref_enc xattn`: the cross-attention reference encoder. Lower error on Dance, Ground and Extreme (tables above)
  for 1.6x the training time. It needs more than 24 GB; on a 24 GB card add `--pi_batch 1024` (G1: 29 min).
- `--interface sparse5`: the policy gets only the pelvis, head, wrist and ankle poses of the reference, no joint
  angles (one command format for a person, a VR tracker set or another robot). A keypoint-to-joint network is fit to
  the library before training and restores the reference-action skip (`--kp_skip`), with an auxiliary joint-angle
  loss (`--kp_aux 1.0`). Export with `tools/export_deploy_onnx.py` (input `ref_kp`).
- `--init <ckpt>`: fine-tune a trained policy on a few clips, e.g. one hard clip before it is deployed. Three
  held-out G1 ground transitions went from 120–415 mm to 24–36 mm in 2k iterations. It does not help on short,
  highly dynamic clips.

`uv run python -m partymimic.train -h` lists the remaining options.

## Evaluate, export, visualize

```bash
# sim2real leaderboard protocol (ONNX + deploy YAML, MuJoCo CPU sim2sim on the four MotionDecode splits)
SIM2REAL=<checkout of github.com/EGalahad/sim2real> bash scripts/eval.sh logs/g1_s0/ckpt_012000.pt eval/g1_s0 g1
# the Extreme set (any any4hdmi dataset directory)
SET=data/testsets/extreme_g1 SIM2REAL=... bash scripts/eval.sh logs/g1_s0/ckpt_012000.pt eval/g1_s0 g1
# side-by-side videos of the median clip per split: robot solid, reference ghost, both at global poses
MUJOCO_GL=egl python tools/sim2sim_video.py eval/g1_s0 g1 <robot MJCF> videos --sim2real <sim2real checkout>
```

Apply [`tools/sim2real.patch`](tools/sim2real.patch) to the sim2real checkout first (`git apply`, tested on its
commits `287ac15` and `4a777c8`). It adds the observation terms the PartyMimic deploy YAML reads
(`supermimic.*`: proprioceptive history, reference joints and root, root orientation history) and, for RP1, the
robot config (limits, armatures, standing pose, joint friction) and RP1's tracked bodies in the metrics; G1 scoring
is unchanged. `scripts/eval.sh ... rp1` points it at `data/rp1_28dof_mjlab` and the RP1 test sets.

Also included: `tools/eval_batch.py` and `tools/eval_metrics.py` for fast in-simulator evaluation on the training
clips, and `tools/export_onnx.py` / `tools/play_onnx.py` for a single-motion ONNX in the BeyondMimic layout.

## Diversity analysis

`tools/diversity` measures how much of a library is distinct motion and how new each test set is
([page](docs/index.html#data)). Each 1-s window is described by heading-invariant root, joint and hand / foot
kinematics, z-scored and projected to 32 PCA dimensions, and the library is clustered into 2,000 motion clusters
(`datacard/`). Per source it reports clips, hours and the effective number of motion types (exponential of the
cluster entropy); for the cumulative mixes, the share of each test set's windows that lie within the median
cross-source distance of a training window.

```bash
uv run --with scikit-learn python tools/diversity/analyze.py work/div out/div \
  --source lafan:LAFAN1=data/motions/g1/lafan --source bones:BONES-SEED=data/motions/g1/bones \
  --source kungfu:KungfuAthlete=data/motions/g1/kungfu \
  --md <sim2real>/datasets/motiondecode --test Extreme=data/testsets/extreme_g1
uv run --with scikit-learn python tools/diversity/figure.py work/div out/div out/div/data.png
```

## Speed on LAFAN (XMimic's benchmark)

[XMimic](https://github.com/nico-bohlinger/xmimic)'s protocol on our RTX 4090s: all runs train on its 40 LAFAN1 clips
(SHA-256 checked) and are scored on the same clips, 4 replicas every 100 iterations, 3 seeds. Minutes to
BlinkMimic's target; compilation excluded by XMimic's rule, added back in the start-up column.

| Method | GPUs | it/s | Return ≥ 0.82 | + 0 failed, held ×3 | Held, with start-up | Return at 3,500 it |
|---|---:|---:|---:|---:|---:|---:|
| BlinkMimic (main) | 1 | 7.8 | 2.93 | 3.45 | 3.62 | 0.865 |
| XMimic | 1 | 9.3 | 2.32 | 3.71 | 3.96 | 0.860 |
| **PartyMimic** | 1 | 8.2 | 2.45 | 5.17 | 5.57 | **0.883** |
| XMimic (actor \| learner) | 2 | 13.2 | 1.71 | **2.89** | **3.17** | 0.863 |
| **PartyMimic** (`--pipeline`) | 2 | 12.7 | **1.66** | 3.41 | 3.82 | **0.883** |

PartyMimic reaches return 0.82 in the fewest iterations and ends highest. It is slower to zero failures: one or two
hard training clips keep failing for a while; BlinkMimic and XMimic search for those clips with evaluation probes,
which PartyMimic dropped because they did not improve held-out tracking.

## The recipe

What matters most, measured by the change in G1 held-out Loc/Man error when a factor is removed (ablations on an
earlier 9,173-clip library; the [page](docs/index.html#recipe) explains each factor):

| Factor | Setting | Effect when removed |
|---|---|---|
| Muon for policy and world model | `--pi_opt muon --wm_opt muon` | Adam instead: +10.7 mm |
| Narrow tracking-reward kernels | `--std_scale 0.35` | +6.5 mm |
| 32-step imagination | `--horizon 32` | 16 steps: +5.9 mm |
| Reference-action skip | `--pi_ref_skip` | +5.6 mm |
| Policy EMA | `--pi_ema 0.995` | +3.4 mm |
| Evaluator's collision model | `--robot g1mj` | LAFAN falls 14/40 instead of 3/40 |
| Value-gradient consistency | `--v_gcons 0.5` | +1.0 mm |
| Diversity-balanced resets | `--div_k 24` | Ground +6–19 mm |
| Self-imitation | `--sil 1` | +0.5 mm early in training |

## Attribution and license

PartyMimic is derived from [BlinkMimic](https://github.com/fuyichen0725/blinkmimic), which builds on
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) and
[InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack). All three are MIT-licensed; their notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Methods: SuperMimic (Liao et al., arXiv:2610.28480, 2026), SHAC (Xu et al., 2022), BeyondMimic (Liao et al., 2025)
and Muon (Jordan et al., 2024).
Simulation: [mjlab](https://github.com/mujocolab/mjlab) and [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).
Retargeting: [human-humanoid-tools](https://github.com/Roboparty/human-humanoid-tools) (Apache-2.0).
Robots: Unitree G1 ([unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), BSD-3-Clause) and RoboParty RP1
([rp1_description](https://github.com/Roboparty/rp1_description)).

Code: MIT ([LICENSE](LICENSE)). The checkpoints, videos and built motion libraries are derived from the motion
datasets above and follow their licenses (LAFAN1: CC BY-NC-ND 4.0; BONES-SEED: its own license; MotionDecode:
non-commercial terms); they are not covered by the MIT License.
