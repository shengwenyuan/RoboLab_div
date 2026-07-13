# RoboLab Robotiq 2F-85 simulation asset

The canonical standalone model is `../robotiq_2f_85.urdf`. The UR5e assembly
`../ur5e_robotiq_2f_85.urdf` is generated from it and the unchanged
`../ur5e_mesh.urdf`:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/build_ur5e_robotiq_2f85_urdf.py
/workspace/isaaclab/_isaac_sim/python.sh scripts/build_ur5e_robotiq_2f85_urdf.py --check
```

Do not hand-edit the generated assembly.

## Model contract

- Geometry, body frames, and inertia come from MuJoCo Menagerie
  `robotiq_2f85_v4` at commit
  `71f066ad0be9cd271f7ed58c030243ef157af9f4`. v4 was rebuilt from
  Robotiq STEP CAD; its STL coordinates are already metres (`scale=1`).
- The v4 root is the physical installation frame: `Z=0` is the tool mounting
  plane and jaws open along `+/-X`. The generated Berkeley assembly uses
  `tool0 -> gripper_root = [0, 0, 0] m, Rz(+pi/2)`, retaining RoboLab's
  previous jaw-opening direction without inventing an axial spacer.
- `base_coupling.stl` is visual and collision geometry. It spans
  `Z=-3.0...+13.9 mm`; the negative skirt wraps the mounting interface. The
  upstream v4 model omitted its collision geometry, so RoboLab explicitly
  adds the matching mesh collider; Isaac imports it with the same convex-hull
  policy as the UR5e.
- Part masses sum to `0.89999986 kg`. The symmetric base COM is calibrated so
  the complete open gripper matches Robotiq's published `0.900 kg` and
  axial COM `57 mm` from the mounting plane. Every moving part retains the v4
  principal inertia.
- `finger_joint` is the only commanded joint. Five joints mimic it with
  multiplier `+1`; their link frames are the v4 frames, so the old ROS `-1`
  inner-finger sign must not be reused. Against the exact v4 equality solution,
  the nominal mimic trajectory has at most `1.64 um` pad-position error. The
  Isaac importer is configured with zero mimic natural frequency and damping
  ratio, selecting a hard PhysX constraint instead of its very soft default.
- All six simplified mimic coordinates use the physical motion interval
  `[0, 0.9] rad`. Menagerie's negative spring/follower solver ranges are only
  needed by its equality-constrained closed loops; retaining them in a linear
  mimic model lets contact loads spread the fingers past their open stop.
  RoboLab's closed command is
  `0.8 / (2*0.485) = 0.824742268 rad`, reproducing v4's split-tendon balance.
  Pad collision-surface gap is approximately `86.878 mm` open, `2.726 mm` at
  `q=0.8`, and `-0.093 mm` at the commanded close in collision-free kinematics.
  With articulation self-collision active and the UR5e's default convex-hull
  import policy preserved, PhysX stops at the physical pad/inner-finger
  contact manifold (approximately zero pad-surface gap, within solver
  tolerance) instead of allowing that nominal preload to interpenetrate.
  `q=0.9` is never commanded because it passes the closure singularity.
- Each pad uses two `4 x 22 x 18.75 mm` contact boxes, each `1.75 g`, plus the
  CAD pad mesh for visual inspection. Isaac expands only the two tiny pad
  collision instance roots, then binds the four boxes to the same spatial
  friction split as Menagerie: lower `static=dynamic=0.7`, upper
  `static=dynamic=0.6`, both with zero restitution. The master effort cap is
  `5 N m` rather than the previous `1000 N m` placeholder.

## TCP contract

No TCP rigid body is added to the physical URDF. RoboLab keeps four concepts
separate:

- UR manufacturer frame: `tool0`.
- Berkeley dataset operational frame: fixed `tool0 + 170 mm` along local Z.
- Robotiq 2018 manual nominal TCP: `tool0 + 171 mm` for that documented setup.
- Moving pad center: diagnostic geometry, never a robot TCP.

The Berkeley frame is injected as a massless Pinocchio operational frame and
resolved from the observed `tool0` pose in the Cosmos client. The bare
six-DOF UR5 description and legacy `tool0` profile remain unchanged.

## Physics boundary

URDF/PhysX mimic gives a reliable nominal parallel-pinch model, but does not
carry v4's equality-constraint forces, spring compliance, or tendon load
sharing. True adaptive/encompassing contact requires a combined UR5+gripper
MJCF/USD import that demonstrably preserves the two loop closures, driver
equality, tendon, and contact exclusions.

Articulation self-collision remains enabled. A custom URDF spawner authors only
the six Menagerie mechanism exclusions plus two fixed mounting-neighbour
exclusions (`gripper base <-> flange/wrist_3`) required because the coarse UR5
collision proxies overlap the bolted coupling. All other arm, gripper, object,
and table collision pairs remain active.

The production `UR5eCfg` retains RoboLab's historical gravity-disabled robot
default so existing UR5 tasks do not silently change control behavior. The
dedicated validator opts into gravity and feeds forward PhysX's articulated
gravity-compensation torques, so the calibrated `0.9 kg` payload is exercised
without measuring controller sag. It separately records jaw self-contact and
four body-to-table filtered contacts, then requires closure/reopen, mount
stability, slow-descent contact, sustained hold, retract clearance, and finite
traces to pass.
