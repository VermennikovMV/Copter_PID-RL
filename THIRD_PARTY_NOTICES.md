# Third-party notices

Author-owned project code and weights have no license grant assigned by this export.
No whole-project MIT license is asserted. External projects are installed separately.

| Component | License recorded in installed distribution / upstream source | Use |
|---|---|---|
| NumPy | BSD-3-Clause plus bundled 0BSD/MIT/Zlib/CC0 components | inference, metrics |
| PyTorch | BSD-3-Clause | optional training |
| Stable-Baselines3 | MIT | optional PPO training |
| Gymnasium | MIT | environment API |
| PyBullet | Zlib | fast physical proxy |
| pymavlink | LGPLv3 | offline auditor imports, MAVLink |
| Matplotlib | Matplotlib license agreement (PSF-based) | optional plotting |
| pandas | BSD-3-Clause | optional analysis |
| SciPy | BSD-3-Clause with bundled third-party notices | optional analysis |
| pytest | MIT | development dependency |
| Pillow | MIT-CMU | optional images |
| ArduPilot | GPL-3.0-or-later | external SITL; not bundled |
| Gazebo gz-sim | Apache-2.0 | external simulator; not bundled |
| ardupilot_gazebo | LGPL-3.0 (upstream LICENSE.md) | external bridge; not bundled |

Python license values above come from installed package metadata, with exact versions
and full available notices in [third_party](third_party). Original installed notices
are retained byte-for-byte. These dependency licenses do not relicense this research code.

External upstream license sources: [ArduPilot COPYING](https://github.com/ArduPilot/ardupilot/blob/master/COPYING.txt),
[gz-sim LICENSE](https://github.com/gazebosim/gz-sim/blob/gz-sim8/LICENSE),
[ardupilot_gazebo LICENSE.md](https://github.com/ArduPilot/ardupilot_gazebo/blob/main/LICENSE.md).
