#!/bin/bash
# Download the G1-retargeted LAFAN1 motions and the G1 description, then build the MJCF asset; set up RP1 from
# RoboParty's rp1_description (meshes) and assets/rp1_28dof_mjlab (capsule-collision MJCF).
# LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0; the Unitree G1 description is BSD-3-Clause.
set -euo pipefail
cd "$(dirname "$0")/.."
HF=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
API=https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main
list() { curl -s "$API/$1" | python3 -c "import json,sys; [print(x['path']) for x in json.load(sys.stdin)]"; }
mkdir -p data/lafan1_g1 data/g1_description/meshes
for f in $(list g1); do curl -sfL -o "data/lafan1_g1/$(basename "$f")" "$HF/resolve/main/$f"; done
curl -sfL -o data/g1_description/g1_29dof_rev_1_0.urdf "$HF/resolve/main/robot_description/g1/g1_29dof_rev_1_0.urdf"
for f in $(list robot_description/g1/meshes); do curl -sfL -o "data/g1_description/meshes/$(basename "$f")" "$HF/resolve/main/$f"; done
echo "motions: $(ls data/lafan1_g1 | wc -l) clips"
uv run python tools/urdf_to_mjcf.py

RP1=19081d6f9fd5b613a304a7749f5886b30eb7a022  # github.com/Roboparty/rp1_description
mkdir -p data/rp1_28dof_mjlab/meshes
curl -sfL "https://codeload.github.com/Roboparty/rp1_description/tar.gz/$RP1" |
  tar -xz -C data/rp1_28dof_mjlab/meshes --strip-components=2 "rp1_description-$RP1/meshes"
cp assets/rp1_28dof_mjlab/rp1_28dof.xml data/rp1_28dof_mjlab/
echo "RP1: $(ls data/rp1_28dof_mjlab/meshes | wc -l) meshes"

python3 tools/data/n3_assets.py

UM=1a37b051a10be723405b7ed6dc839361af036d88  # github.com/unitreerobotics/unitree_mujoco: G1 of tools/diversity
mkdir -p data/unitree_mujoco_g1
curl -sfL "https://codeload.github.com/unitreerobotics/unitree_mujoco/tar.gz/$UM" |
  tar -xz -C data/unitree_mujoco_g1 --strip-components=3 "unitree_mujoco-$UM/unitree_robots/g1"
