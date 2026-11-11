# Noetix N3 V3

The training model and 30 visual meshes are vendored from the local GR00T-WholeBodyControl
`gear_sonic/data/assets/robot_description/{mjcf/n3.xml,urdf/n3/meshes/}` assets, derived from the
`2026-11-03_n3_current` training snapshot. `provenance.json` records their hashes. Only the MJCF
mesh directory is changed to `meshes`; joint frames, inertia, collision geometry and exclusions
are preserved. These third-party robot assets are not relicensed by PartyMimic's MIT license.

Like RP1, training loads the copy under `data/`, prepared with:

```bash
python tools/data/n3_assets.py
```

This copies `n3_29dof.xml` and `meshes/` to `data/n3_29dof_mjlab/`. After editing the source XML,
run the command again to update the training copy. Both source and runtime XMLs use meshes beside
the XML, so neither depends on an external GR00T checkout. The vendored meshes are ordinary Git
files and can travel with the repository's bundle.

`src/partymimic/n3/n3_constants.py` supplies the implicit PD actuators, friction, default pose and
action scales, matching GR00T's N3 metadata. The MJCF has 29 hinges and two fixed toes. Tracking
uses 15 moving bodies, including `head_yaw_link`; sparse5 uses that actual head, both hand-pitch
links and both ankle-roll links, plus the root.

The flat-foot ankle origin is 0.066 m above the ground (capsule centre at -0.056 m, radius 0.010 m).
Contact rewards use a 0.078 m ankle-height threshold; this is the geometric flat-foot height plus
the same 0.012 m margin as the default G1/RP1 metric. The geometric proxy is not a contact sensor.

Motion inputs remain NPZ (`fps`, `joint_pos`, `body_pos_w`, `body_quat_w`). Named short N3_0905
joints are aliased to V3 names before gathering into MJCF order; already converted DFS data is
not permuted twice. Unnamed files must explicitly declare `joint_order=depth_first` in this
model's hinge order. Input root quaternions are wxyz, positions metres, angles radians. MotionLib
resamples to 50 Hz and recomputes velocities and reference body poses with the V3 model.

This name/order conversion does not retarget old N3_0905 geometry or repair motion quality. Review
V3 foot height, limits and collisions on the reference set before a full run. Keep the separately
stored `seed_unclassified` clips out of the chosen `--motion_dir` unless intentionally testing them.
